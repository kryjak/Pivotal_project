#!/bin/bash

# 1. Check Linux distribution and ask user about editor installation
install_dependencies() {
    local package_manager
    local install_command

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            fedora)
                package_manager="dnf"
                install_command="sudo dnf install -y"
                ;;
            ubuntu|debian)
                package_manager="apt-get"
                install_command="sudo apt-get install -y"
                ;;
            centos|rhel)
                package_manager="yum"
                install_command="sudo yum install -y"
                ;;
            *)
                echo "Unsupported distribution: $ID"
                return 1
                ;;
        esac
    else
        echo "Unable to determine the distribution"
        return 1
    fi

        # Check if Python 3.10 is available
    if command -v python3.10 &>/dev/null; then
        echo "Python 3.10 is available. Installing python3.10-venv..."
        $install_command python3.10-venv
    else
        echo "Warning: Python 3.10 is not available on this system. Please install Python 3.10 before proceeding."
        return 1
    fi

    while true; do
        read -p "Do you want to install an editor? (vim/1, emacs/2, or no/0): " editor_choice
        case $editor_choice in
            vim|VIM|Vim|VI|Vi|vi|1)
                if [ "$package_manager" = "apt-get" ]; then
                    sudo apt-get update
                fi
                $install_command vim
                break
                ;;
            emacs|EMACS|Emacs|2)
                if [ "$package_manager" = "apt-get" ]; then
                    sudo apt-get update
                fi
                $install_command emacs
                break
                ;;
            no|NO|No|0)
                echo "Skipping editor installation."
                break
                ;;
            *)
                echo "Invalid choice. Please enter vim/1, emacs/2, or no/0."
                ;;
        esac
    done
}

# Call the installation function first
if ! install_dependencies; then
    echo "Error during dependency installation. Please fix the issues and run the script again."
    exit 1
fi

final_directory=""

# 2. Ask if user wants to create a new folder
create_folder() {
    while true; do
        read -p "Do you want to create a new folder? (yes/1 or no/2): " create_choice
        case $create_choice in
            yes|YES|Yes|1)
                read -p "Enter the path for the new folder (press Enter for default 'gitrepos'): " folder_path
                if [ -z "$folder_path" ]; then
                    folder_path="gitrepos"
                fi
                mkdir -p "$folder_path"
                final_directory="$folder_path"
                echo "Created directory: $folder_path"
                return 0
                ;;
            no|NO|No|2)
                echo "Proceeding without creating a new folder."
                return 1
                ;;
            *)
                echo "Invalid choice. Please enter yes/1 or no/2."
                ;;
        esac
    done
}

create_folder

# 3. Ask user for git clone method and clone the repository
clone_repo() {
    local repo_url
    while true; do
        read -p "Do you want to clone using SSH (1) or HTTPS (2)? " choice
        case $choice in
            1|ssh|SSH)
                repo_url="git@github.com:kryjak/Pivotal_project.git"
                break
                ;;
            2|https|HTTPS)
                repo_url="https://github.com/kryjak/Pivotal_project.git"
                break
                ;;
            *)
                echo "Invalid choice. Please enter 1 for SSH or 2 for HTTPS."
                ;;
        esac
    done

    if [ -n "$final_directory" ]; then
        git -C "$final_directory" clone "$repo_url"
    else
        git clone "$repo_url"
    fi
    final_directory="${final_directory:+$final_directory/}Pivotal_project"
}

clone_repo


echo "Done!"
echo "To finish the setup, please run the following commands:"
echo "cd $final_directory"
echo "python3.10 -m venv .py310_venv"
echo "source .py310_venv/bin/activate"
echo "pip install -r requirements.txt"
echo "python3.10 -m ipykernel install --user --name=custom_venv --display-name='Custom Virtual Env'"
echo "-------------------"
echo "You might have to reload your editor window in order for the editor to recognise the new kernel."
