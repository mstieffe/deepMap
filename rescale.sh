input_path=./data/reference_snapshots/dodecane_cg
target_file=./data/reference_snapshots/dodecane_cg_mapped/0.gro

rescale_path=./data/reference_snapshots/dodecane_cg_rescaled

cp -r $input_path $rescale_path


tag=$( tail -n 1 $target_file )
arr=(${tag// / })
y1=${arr[0]}  
y2=${arr[1]}  
y3=${arr[2]}  

echo $y1
echo $y2
echo $y3

for file in $rescale_path/*;
do
tag=$( tail -n 1 $file )
arr=(${tag// / })
x1=${arr[0]}  
x2=${arr[1]}  
x3=${arr[2]} 

s1=$(echo "scale=5 ; $y1 / $x1" | bc)
s2=$(echo "scale=5 ; $y2 / $x2" | bc)
s3=$(echo "scale=5 ; $y3 / $x3" | bc)

gmx editconf -f $file -o $file -scale $s1 $s2 $s3

done

cd $rescale_path
rm *#*


