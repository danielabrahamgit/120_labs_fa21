import numpy as np

def fast_get_fourier_coeffs(x):
	N = len(x)
	ak = np.zeros(N//2 + 1)
	ak = np.fft.fft(x)[:N//2 + 1] / len(x)
	return ak

def fast_recon(k_sig, rk_sig, N):
	ak_new = np.zeros(N, dtype=rk_sig.dtype)
	ak_new[k_sig] = rk_sig
	ak_new[-(N//2):] = np.flip(np.conj(ak_new[1:(N//2)+1]))
	return np.real(np.fft.ifft(ak_new) * N)


def test_get_fourier_coeffs(test_func):
	x1 = np.array([1, 0, 0, 0, 0, 0])
	x2 = np.array([1, 1, 1, 1])
	x3 = np.array([-9, 17, 4, 1, 0, 8, 2, -12, 19, 16, -10])
	x4 = np.array([-6, 5, -20, -13, 18, -2, 18, -9, 6, 12, 12, -19, 5, 16])
	predef_tests = [x1, x2, x3, x4]
	
	
	# Pre-defined test cases
	for i, x in enumerate(predef_tests):
		expected = fast_get_fourier_coeffs(x)
		actual = test_func(x)
		if np.linalg.norm(actual - expected) > 1e-6:
			print(f'Test{i+1} Failed.')
			print(f'x = {x}')
			print('Expected:', expected)
			print('Actual:', actual)
			return
	
	# randomized test cases
	for N in range(16, 25):
		x = np.random.normal(0, 4, N)
		if np.linalg.norm(test_func(x) - fast_get_fourier_coeffs(x)) > 1e-6:
			print('Random Signal Test Failed')
			return
	print('All Tests Passed!')

def test_M_most_sig(test_func):
	xz = np.zeros(10)
	k, ak = test_func(xz, 5)
	if len(k) > 0 or len(ak) > 0:
		print('Test Failed: Zeros are not significant')
		return
	xz[7] = 1
	k, ak = test_func(xz, 10)
	if len(k) != 1 or len(ak) != 1:
		print('Test Failed: Zeros are not significant!')
		return
	
	x1 = np.array([0, 10, 12, 0])
	x2 = np.array([1 -100j, 0.1 - 0.2j, 0, 0, 200j])
	x3 = np.array([-12, -12, -12, 10, 10, 10])
	tests = [(x1, 2), (x1, 1), (x2, 5), (x2, 7), (x2, 2), (x3, 3), (x3, 6)]
	
	tests_expected = [
				(np.array([2, 1]), np.array([12, 10])),
				(np.array([2]), np.array([12])),
				(np.array([4, 0, 1]), np.array([0. +200.j , 1. -100.j , 0.1  -0.2j])),
				(np.array([4, 0, 1]), np.array([0. +200.j , 1. -100.j , 0.1  -0.2j])),
				(np.array([4, 0]), np.array([0.+200.j, 1.-100.j])),
				(np.array([0, 1, 2]), np.array([-12, -12, -12])),
				(np.array([0, 1, 2, 3, 4, 5]), np.array([-12, -12, -12,  10,  10,  10]))
	]
	
	for i, test in enumerate(tests):
		x, M = test
		act_k, act_ak = test_func(x, M)
		exp_k, exp_ak = tests_expected[i]
		if len(exp_k) != len(act_k) or np.linalg.norm(np.sort(exp_k) - np.sort(act_k)) > 1e-6 \
            or len(exp_ak) != len(act_ak) or np.linalg.norm(np.sort(exp_ak) - np.sort(act_ak)) > 1e-6:
			print(f'Test{i+1} Failed.')
			print(f'x = {x}')
			print(f'M = {M}')
			print(f'Expected Indices = {exp_k}')
			print(f'Expected Values = {exp_ak}')
			print(f'Actual Indices = {act_k}')
			print(f'Actual Values = {act_ak}')
			return
	print('All Tests Passed!')

def test_chop(test_func):
	for N in range(99, 200):
		for block_size in range(10, 20):
			x = np.random.normal(0, 5, N)
			Nz = int(np.ceil(len(x) / block_size) * block_size) - len(x)
			x_padded = np.concatenate((x, np.zeros(Nz)))
			blocks = test_func(x_padded, block_size)
			x_recon = np.concatenate(blocks)
			if np.linalg.norm(x_padded - x_recon) > 1e-6:
				print('Test Failed')
				return
	
	print('All Tests Passed!')
			