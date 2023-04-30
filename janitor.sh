janitor() {
	for dir in $(find SEM*); do
		if [[ -d $dir ]]; then
			for file in $(ls $dir -a); do
				if [[ $file == *".vscode"* || $file == *".git"* || $file == *".gitignore"* || $file == *".github"* || $file == *"__pycache__"* || $file == *".idea"* || $file == *".aux" || $file == *".log" || $file == *".synctex.gz" || $file == *".out" || $file == *".fdb_latexmk" || $file == *".fls" || $file == *".clang-format" ]]; then
					rm -rvf $dir/$file
				fi
			done
		fi
	done
}

janitor
