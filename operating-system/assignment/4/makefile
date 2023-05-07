all:
	gcc traffic.c -Wall -lpthread -o traffic
	echo Running first test case
	./traffic INPUT.txt
	echo Running second test case
	./traffic INPUT2.txt
	rm ./traffic

built:
	gcc traffic.c -Wall -lpthread -o traffic

rebuilt:
	gcc traffic.c -Wall -lpthread -o traffic

run:
	./traffic INPUT.txt
	./traffic INPUT2.txt

clean:
	rm ./traffic
