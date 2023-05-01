# write a script that would remove space with underscore in all the filenames in the current directory

for file in *; do
	if [[ -d $file ]]; then
		cd $file
		for file2 in *; do
			if [[ -f $file2 ]]; then
				mv "$file2" "${file2// /_}"
				echo "$file2"
			fi
		done

		cd ..
	fi
done
