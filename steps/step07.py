import numpy as np
import unittest

# addなどでその後の関数を複数回呼ばないようにする

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError("{} is not supported ".format(type(data)))
		self.data = data
		self.grad = None
		self.creator = None
		self.generation = 0

	def set_creator(self, func):
		self.creator = func
		self.generation = func.generation +1

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)
		
		funcs = []
		seen_set = set()

		def add_func(f):
			print(f)
			if f not in seen_set:
				print("not in seenset")
				print("seennset",seen_set,"function",f)
				funcs.append(f)
				seen_set.add(f)
				funcs.sort(key=lambda x: x.generation)
			else:
				print("in seen set")
				print("not in seenset")
				print("seennset",seen_set,"function",f)
			print("  ")
		add_func(self.creator)
		print("functionlist first",funcs)	
			

		while funcs:
			f = funcs.pop()
			gys = [output.grad for output in f.outputs]
			#print(gys,f)
			gxs = f.backward(*gys)
			#print(gxs,type(gxs))
			if not isinstance(gxs, tuple):
				gxs = (gxs,)
			#print("function",f,"grad",gxs)
			for x, gx in zip(f.inputs, gxs):
				if x.grad is None:
					x.grad = gx
				else:
					x.grad = x.grad + gx
				if x.creator is not None:
					#print("================")
					#print("cre",x.creator)
					funcs.append(x.creator)
					#add_func(x.creator)
			print("functionlist",funcs)	
			print("")
def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys,)
		outputs = [Variable(as_array(y)) for y in ys]
		self.generation = max([x.generation for x in inputs])
		for output in outputs:
			output.set_creator(self)
		self.inputs = inputs
		self.outputs = outputs
		return outputs if len(outputs)>1 else outputs[0]

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy):
		raise NotImplementedError()

def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data-eps)
	x1 = Variable(x.data + eps)
	y0 = f(x0)
	y1 = f(x1)
	return (y1.data - y0.data) / (2*eps)

class SquareTest(unittest.TestCase):
	def test_forward(self):
		x = Variable(np.array(2.0))
		y = square(x)
		expected = np.array(4.0)
		self.assertEqual(y.data, expected)
	def test_backward(self):
		x = Variable(np.array(3.0))
		y = square(x)
		y.backward()
		expected = np.array(6.0)
		self.assertEqual(x.grad, expected)
	
	def test_gradient_check(self):
		x = Variable(np.random.rand(1))
		y = square(x)
		y.backward()
		num_grad= numerical_diff(square, x)
		flg = np.allclose(x.grad, num_grad)
		self.assertTrue(flg)
class Square(Function):
	def forward(self, x):
		y = x ** 2
		return y

	def backward(self, gy):
		x = self.inputs[0].data
		gx = 2 * x * gy
		return gx
class Exp(Function):
	def forward(self, x):
		y = np.exp(x)
		return y

	def backward(self, gy):
		x = self.input.data
		gx = np.exp(x) * gy
		return gx

class Add(Function):
	def forward(self, x0, x1):
		y = x0 + x1
		return y
	def backward(self, gy):
		return gy, gy
def square(x):
	return Square()(x)

def exp(x):
	return Exp()(x)

def add(x0,x1):
	return Add()(x0,x1)

x = Variable(np.array(2.0))
a = square(x)
y = add(a, a)
y.backward()

print(y.data)
print(x.grad)