from vllm import LLM, SamplingParams
import re
import json

class vllm_Socket():
    def __init__(self, configs):
        self.model = configs.model
        self.temperature = configs.temperature if configs.temperature is not None else 0.7
        self.top_p = configs.top_p if configs.top_p is not None else 0.95
        self.max_tokens = configs.max_tokens if configs.max_tokens is not None else 1024
        self.first_prompt = configs.first_prompt
        self.retry_prompt = configs.retry_prompt
        self.retry = configs.retry
        self.force_retry = configs.force_retry

        self.load_templates()
        self.load_model()

    def load_templates(self):
        with open(self.first_prompt, 'r') as f:
            self.first_prompt = f.read()
        with open(self.retry_prompt, 'r') as f:
            self.retry_prompt = f.read()
    
    def load_model(self):
        self.llm = LLM(model=self.model, trust_remote_code=True)
        print("[Info] Successfully load Time-R1 using vllm")

    def process_instance(self, instance):
        # instance: seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
        x_ts = instance[0].squeeze().tolist()
        y_ts = instance[1].squeeze().tolist()

        # x_timestamp = instance[2].tolist()
        y_timestamp = instance[3].tolist()

        x_dy = instance[4]
        y_dy = instance[5]

        channel_info = instance[-1]
        dataset_info = instance[-2]

        return x_ts, y_ts, x_dy, y_dy, y_timestamp, channel_info, dataset_info
    
    def parse_model_output(self, text: str) -> tuple[str | None, str | None]:
        pattern = re.compile(r"<think>(.*?)</think>\s*<answer>\s*```(.*?)```\s*</answer>", re.DOTALL)
        match = pattern.search(text)
        if match:
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()
            return think_content, answer_content
        else:
            return None, None
    
    def _format_result(self, answer_content: list, data_instance: tuple) -> dict:
        x_ts = data_instance[0].squeeze().tolist()
        y_ts = data_instance[1].squeeze().tolist()
        x_timestamp = data_instance[2].tolist()
        y_timestamp = data_instance[3].tolist()
        x_dy = data_instance[4]
        y_dy = data_instance[5]
        hetero_x_time = data_instance[6]
        hetero_y_time = data_instance[7]

        try:
            if answer_content:
                pred_formatted = [[ts, answer_content[i]] for i, ts in enumerate(y_timestamp)]
            else:
                raise ValueError("answer_content is empty after inference.")
        except Exception:
            pred_formatted = [[ts, -1] for ts in y_timestamp]
            raise ValueError("Failed to format the prediction results.")

        x_table_formatted = [(x_timestamp[i], x_ts[i]) for i in range(len(x_ts))]
        x_dy_table_formatted = [(hetero_x_time[i], x_dy[i]) for i in range(len(x_dy))]
        y_dy_table_formatted = [(hetero_y_time[i], y_dy[i]) for i in range(len(y_dy))]
        y_table_formatted = [(y_timestamp[i], y_ts[i]) for i in range(len(y_ts))] # Ground truth

        num_channels = 1
        if x_ts and isinstance(x_ts[0], (list, tuple)):
            num_channels = len(x_ts[0])
            
        y_timestamp_with_placeholders = []
        for i, ts in enumerate(y_timestamp):
            if num_channels == 1:
                placeholders = f'<your_prediction_value[{i}]>'
            else:
                placeholders = [f'<your_prediction_value[{i}][{j}]>' for j in range(num_channels)]
            y_timestamp_with_placeholders.append([ts, placeholders])

        return {
            'pred': pred_formatted, 
            'x_table': x_table_formatted,
            'x_dy_table': x_dy_table_formatted,
            'y_timestamp': y_timestamp_with_placeholders,
            'y_dy_table': y_dy_table_formatted,
            'y_table': y_table_formatted,
        }

    def __call__(self, data_instance):
        retry = self.retry  # create a local copy to ensure self.retry remains unchanged
        # instance: ts_x, ts_y, tm_x, tm_y, (dy_tm_x, general, channel, [dy_x]), (dy_tm_y, general, channel, [dy_y])

        x_ts, y_ts, x_dy, y_dy, y_timestamp, channel_info, dataset_info = self.process_instance(data_instance)

        # format prompt
        first_prompt = self.first_prompt.format(hetero_general=dataset_info,
                                                hetero_channel=channel_info,
                                                seq_len=len(x_ts),
                                                x=x_ts,
                                                batch_x_hetero=x_dy,
                                                batch_y_hetero=y_dy,
                                                pred_len=len(y_ts))
        retry_prompt = first_prompt

        sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        
        if self.force_retry:
            for i in range(retry):
                outputs = self.llm.generate
                pass
        else:
            while True:
                if retry == 0:
                    print("[Warning]: Reach max retry num, stopping...")
                    break
                try:
                    outputs = self.llm.generate(first_prompt, sampling_params)
                    generated_text = outputs[0].outputs[0].text
                    
                    think_content, answer_content = self.parse_model_output(generated_text)
                    
                    if think_content is None or answer_content is None:
                        raise AssertionError("[Warning]: Failed to parse the output according to the format")
                    
                    answer_content = eval(answer_content)  # convert string representation of list to actual list

                    break
                except:
                    retry -= 1
                    pass
        
        result = self._format_result(answer_content, data_instance)
        
        return result, first_prompt
