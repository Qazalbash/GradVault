# write a shell that will recursively go into each directory and delete all files that have name
# .vscode
# .git
# .gitignore
# .github
# __pycache__
# .idea

# 1. find all files with the above names

# 2. delete them

# 3. if the file is a directory, go into it and repeat the process

# start writing

# it take one argument as the path to the directory
janitor() {
	for file in $(find SEM*); do
		if [[ $file == *".vscode"* || $file == *".git"* || $file == *".gitignore"* || $file == *".github"* || $file == *"__pycache__"* || $file == *".idea"* ]]; then
			echo "deleting $file"
			rm -rvf $file
		fi
	done
}

janitor
