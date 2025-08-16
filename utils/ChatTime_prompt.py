GEN_SYS_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PRE_SYS_PROMPT = "You are a helpful assistant that performs time series prediction. The user will provide a sequence and you will predict the sequence."
ANA_SYS_PROMPT = "You are a helpful assistant that performs time series analysis. The user will provide a sequence and you will respond to the questions based on this sequence."

PRE_INST_PROMPT = "Please predict the following sequence carefully."
PRE_INST_PROMPT_TEXT = "Please predict the following sequence carefully. Context knowledge you may consider: {}"
ANA_INST_PROMPT_TEXT = "Please answer the following question carefully after analyzing the sequence: {}"

TEMPLATE = """{}

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def getPrompt(flag, instruction=None, input=None, response=None, context=None):
    if flag == "general":  # instruction, [input], [response]
        system = GEN_SYS_PROMPT
        if instruction is None:
            raise ValueError("Instruction must be provided for general tasks.")
        else:
            instruction = instruction
        input = "" if input is None else input
        response = "" if response is None else response

    elif flag == "prediction":  # [context], input, [response]
        system = PRE_SYS_PROMPT
        instruction = PRE_INST_PROMPT if context is None else PRE_INST_PROMPT_TEXT.format(context)
        if input is None:
            raise ValueError("Input must be provided for prediction tasks.")
        else:
            input = input
        response = "" if response is None else response

    elif flag == "analysis":  # instruction, input, [response]
        system = ANA_SYS_PROMPT
        if instruction is None:
            raise ValueError("Instruction must be provided for analysis tasks.")
        else:
            instruction = ANA_INST_PROMPT_TEXT.format(instruction)
        if input is None:
            raise ValueError("Input must be provided for analysis tasks.")
        else:
            input = input
        response = "" if response is None else response

    else:
        raise ValueError("Flag must be one of 'general', 'prediction', or 'analysis'.")

    prompt = TEMPLATE.format(system, instruction, input, response)

    return prompt
