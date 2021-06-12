# time.time() 耗时测试

~~~python
tic = time.time()
N = 100
for i in range(N):
	......
average_infer_time = (time.time() - tic) / N
~~~

