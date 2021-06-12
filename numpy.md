# np.expand_dims

~~~python
>>> a_np
array([[ 0.991783  , -1.4756904 ,  0.21210898,  0.8881568 ,  2.8987153 ],
       [-0.7357134 , -0.06555091, -0.7445159 , -0.27188963,  1.3269119 ]],
      dtype=float32)
>>> bboxes = a_np[:, :4].copy()
>>> lables = a_np[:, -1].copy()

>>> bboxes
array([[ 0.991783  , -1.4756904 ,  0.21210898,  0.8881568 ],
       [-0.7357134 , -0.06555091, -0.7445159 , -0.27188963]],
      dtype=float32)
>>> lables
array([2.8987153, 1.3269119], dtype=float32)

>>> lables = np.expand_dims(lables, 1)
>>> lables
array([[2.8987153],
       [1.3269119]], dtype=float32
~~~

