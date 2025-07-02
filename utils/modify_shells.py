import os
import sys

def modify_sh_files_in_directory(root_dir):
    """
    Traverse a specified directory and its subdirectories to modify specific content 
    in all .sh files.

    :param root_dir: The root directory path to traverse.
    """
    # Define pairs of strings to find and replace.
    # Note: The strings must be an exact match, including spaces and quotes.
    replacements = {
        "--data netztransparenz": "--data California_ISO",
        "--data_config './data_configs/netztransparenz.yaml'": "--data_config './data_configs/California_ISO/fullCAISO.yaml'"
    }
    
    # Count the number of modified files.
    modified_files_count = 0

    print(f"Starting to search for .sh files in directory '{root_dir}'...")

    # os.walk recursively traverses the directory tree.
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file is a .sh file.
            if filename.endswith('.sh'):
                file_path = os.path.join(dirpath, filename)
                
                try:
                    # Open the file in read mode and read its content.
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    original_content = content
                    
                    # Perform the replacement for each pair of strings.
                    for old_str, new_str in replacements.items():
                        content = content.replace(old_str, new_str)

                    # If the content has changed, write it back to the file.
                    if content != original_content:
                        print(f"Modifying file: {file_path}")
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(content)
                        modified_files_count += 1
                    # else:
                        # If no modification is needed, you can handle it silently or print a message.
                        # print(f"File {file_path} does not need modification.")

                except Exception as e:
                    print(f"An error occurred while processing file {file_path}: {e}", file=sys.stderr)

    print("\n--------------------")
    print(f"Processing complete! A total of {modified_files_count} files were modified.")


if __name__ == "__main__":
    # Get the target folder path from user input.
    target_folder = input("Please enter the path to the directory to be scanned: ")

    # Check if the path exists and is a directory.
    if not os.path.isdir(target_folder):
        print(f"Error: The path '{target_folder}' is not a valid directory or does not exist.")
    else:
        # --- Security Warning ---
        print("\n" + "="*55)
        print("WARNING: This script will directly modify .sh files in your folder.")
        print("IT IS STRONGLY RECOMMENDED TO BACK UP YOUR DATA BEFORE RUNNING!")
        print("="*55 + "\n")
        
        # Ask the user for confirmation.
        confirm = input("Are you sure you want to continue? (Type y or yes to proceed): ")
        
        if confirm.lower() in ['y', 'yes']:
            modify_sh_files_in_directory(target_folder)
        else:
            print("Operation cancelled.")