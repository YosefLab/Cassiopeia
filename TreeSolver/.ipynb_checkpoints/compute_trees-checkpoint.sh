for i in 1 2 9
do echo $i
	python run_ivlt_solver.py $i 1> ${i}.out 2> ${i}.err
done
