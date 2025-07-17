def keep_last_third(input_file, output_file, ratio=20):

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    

    total_chars = len(content)
    chars_to_keep = total_chars // ratio
    if total_chars % ratio != 0:
        chars_to_keep += 1
    

    last_third = content[-chars_to_keep:]
    

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(last_third)
    
    print(f"file: {input_file}")
    print(f"origin: {total_chars}")
    print(f"remained: {chars_to_keep}")
    print(f"saved to: {output_file}")


if __name__ == "__main__":
    input_log = "original.log"
    output_log = "trimmed.log"
    ratio = 20
    
    keep_last_third(input_log, output_log, ratio=ratio)