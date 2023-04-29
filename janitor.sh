janitor() {
	for dir in $(find SEM*); do
		if [[ -d $dir ]]; then
			# cd $dir
			echo $(ls -a $dir)
			for file in $(ls $dir -a); do
				if [[ $file == *".vscode"* || $file == *".git"* || $file == *".gitignore"* || $file == *".github"* || $file == *"__pycache__"* || $file == *".idea"* ]]; then
					rm -rvf $dir/$file
				fi
			done
		fi
	done
}

janitor
