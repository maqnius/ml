for f in *.pdf
do
	echo "Processing $f file.."
	pdfcrop --margins '0 0 0 0' $f $f
done
