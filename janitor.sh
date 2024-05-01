janitor() {
	for dir in $(find *); do
		if [[ -d $dir ]]; then
			for file in $(ls $dir -a); do
				# if there is a tex file in the directory, remove the pdf file with same name
				if [[ $file == *".tex" ]]; then
					rm -rvf $dir/${file%.tex}.pdf
				fi
				if [[ $file == *".vscode"* || $file == *".git"* || $file == *".gitignore"* || $file == *".github"* || $file == *"__pycache__"* || $file == *".idea"* || $file == *".aux" || $file == *".log" || $file == *".synctex"* || $file == *".out" || $file == *".fdb_latexmk" || $file == *".fls" || $file == *".toc" || $file == *".clang-format" ]]; then
					rm -rvf $dir/$file
				fi
			done
		fi
	done
}

janitor
