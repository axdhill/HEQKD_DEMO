from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
import time
import numpy as np
import sys
import socket, pickle
from QKDMsg import QKDMsg

from threading import Lock, Thread




ttag_lock = Lock()

cur_tags = []
cur_frame = []
cur_channels = []


def cascade_k(S, v, k) :
	no_blocks = len(S)/k
	if not (float(no_blocks).is_integer()) :
		raise ValueError('k does not evenly divide bit string length!')
	return S[((v-1)*k):(v*k)]


def cascade_parity(block) :
	return int(np.sum(block) % 2 == 1)

def cascade_parity_of_all_blocks(S, k) :
	no_blocks = int(len(S)/k)
	return [cascade_parity(cascade_k(S, v, k)) for v in range(1, no_blocks)]

def cascade_rand_f(k, length) :
	return np.random.choice(list(range(int(np.ceil(length/k)))), size = length)


def cascade_BdelK(B, K) :
	Kprime = np.mod((np.logical_or(B,K) - np.logical_and(B,K)).astype(int))
	return Kprime


def cascade_apply_f(K, f, i) :
	return K[ [f == i for f in f_i] ]


def cascade_calc_xj(K, f, j) :
	return np.mod(np.sum(cascade_apply_f(K, f, j)), 2)


def cascade_binary(A, B, i) :
	if len(A) == 1 :
		return i 
	a_first_half_parity = cascade_parity(A[0:int(len(A)/2)])
	b_first_half_parity = cascade_parity(B[0:int(len(B)/2)])
    
	#if len(A) % 2 != 0 : # If A has odd length
	if a_first_half_parity == b_first_half_parity :
		return cascade_binary(A[int(len(A)/2):], B[int(len(A)/2):], i + int(len(A)/2))
	else :
		return cascade_binary(A[0:int(len(A)/2)], B[0:int(len(B)/2)], i)
        

def get_ttags_from_file(in_file) :
	in_file.seek(0,2)
	while True:
		line = in_file.readline()
		if not line:
			time.sleep(0.05)
			continue
		yield line


def poll_file(fname) :
	logfile = open(fname, "r")
	loglines = get_ttags_from_file(logfile)
	
	frame = []
	tags = []
	channels = []
	for line in loglines : 
		line_array = line.split(" ")
		frame = int(float(line_array[0]))
		channels = int(float(line_array[1]))
		tag = float(line_array[2])

		ttag_lock.acquire()
		cur_tags.append(tag)
		cur_frame.append(frame)
		cur_channels.append(channels)
		ttag_lock.release()



class HEQKD(FloatLayout) :
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	tags = []
	chs = []
	qber = 0
	qber_str = ""
	name = sys.argv[1]

	if name == "Alice" :
		send_port = 51718
		recv_port = 51719
		send_addr = "192.17.210.71"
		recv_addr = "10.194.7.50"
	else :
		send_port = 51719
		recv_port = 51718
		send_addr = "10.194.7.50"
		recv_addr = "192.17.210.71"
	

	sock.bind((recv_addr, recv_port))

	coin_rate = NumericProperty(0)
	singles_rate = NumericProperty(0)
	qber = NumericProperty(0)
	minimum_N = 5
	basis_map = {0:0, 1:0, 2:1, 3:1}

	

	def _socket_listener(self) :
		while True :
			data, addr = self.sock.recvfrom(2*4096)
			msg = pickle.loads(data)
			print("Received type ", msg.msg_type, " at frame ", str(msg.msg_frame))
			if msg.msg_type == "MATCH_REQ" :
				# Find matching indices and transmit back to source
				match_indices_source, match_indices_me = self.do_matching(pickle.loads(msg.msg_payload), self.tags)

				response = QKDMsg()
				response.msg_frame = msg.msg_frame
				response.msg_payload = pickle.dumps(match_indices_source)
				response.msg_type = "MATCH_RESPONSE"
				self.sock.sendto(pickle.dumps(response), (self.send_addr, self.send_port))

				self.match_indices = match_indices_me
				print("Found matching indices ", self.match_indices)
				values = np.array(self.to_value(self.chs))
				self.key_bits = values[self.match_indices]

				print("My bits are ", self.key_bits)
				self.coin_rate = len(self.match_indices)

			if msg.msg_type == "MATCH_RESPONSE" :
				#print("Received match indices ", pickle.loads(msg.msg_payload))
				self.match_indices = pickle.loads(msg.msg_payload)
				values = np.array(self.to_value(self.chs))
				self.key_bits = values[self.match_indices]
				#print("My bits are ", self.key_bits)
				self.coin_rate = len(self.match_indices)

				# At this point both Alice and Bob have matched keys converted to bit strings
				# which may contain errors
				self.chs = np.array(self.chs)

				bases_for_matched_events = self.to_basis(self.chs[self.match_indices])
				# Alice responds with Sift
				response = QKDMsg()
				response.msg_frame = msg.msg_frame
				response.msg_payload = pickle.dumps(bases_for_matched_events)
				response.msg_type = "SIFT_REQ"
				self.sock.sendto(pickle.dumps(response), (self.send_addr, self.send_port))


			if msg.msg_type == "SIFT_REQ" :
				self.chs = np.array(self.chs)
				sift_indices = self.sift(pickle.loads(msg.msg_payload), self.to_basis(self.chs[self.match_indices]))
				response = QKDMsg()
				response.msg_frame = msg.msg_frame
				response.msg_payload = pickle.dumps(sift_indices)
				response.msg_type = "SIFT_RESPONSE"
				self.sock.sendto(pickle.dumps(response), (self.send_addr, self.send_port))

				self.sift_indices = sift_indices
				self.key_bits_sifted = self.key_bits[self.sift_indices]

				print("My sifted bits are ", self.key_bits_sifted)



			if msg.msg_type == "SIFT_RESPONSE" :
				# print("Received sift indices ", pickle.loads(msg.msg_payload))
				self.sift_indices = pickle.loads(msg.msg_payload)
				values = np.array(self.to_value(self.chs))
				self.key_bits_sifted = self.key_bits[self.sift_indices]
				print("My sifted bits are ", self.key_bits_sifted)

				# Now alice prepares to do parameter estimation
				response = QKDMsg()
				response.msg_frame = msg.msg_frame
				[comp_bits, self.remaining_bits] = self.random_subset(self.key_bits_sifted)

				response.msg_payload = pickle.dumps(comp_bits)
				response.msg_type = "PARAM_ESTIM_REQ"
				self.sock.sendto(pickle.dumps(response), (self.send_addr, self.send_port))
				


			if msg.msg_type == "PARAM_ESTIM_REQ" :
				[comp_bits, self.remaining_bits] = self.random_subset(self.key_bits_sifted)
				self.qber = self.calc_qber(pickle.loads(msg.msg_payload), comp_bits)

				response = QKDMsg()
				response.msg_frame = msg.msg_frame
				response.msg_payload = pickle.dumps(self.qber)
				response.msg_type = "PARAM_ESTIM_RESPONSE"
				self.sock.sendto(pickle.dumps(response), (self.send_addr, self.send_port))

				print("QBER: ", self.qber)


			if msg.msg_type == "PARAM_ESTIM_RESPONSE" :
				self.qber = pickle.loads(msg.msg_payload)

				# response = QKDMsg()
				# response.msg_frame = msg.msg_frame
				# response.msg_payload = pickle.dumps(self.qber)
				# response.msg_type = "PARAM_ESTIM_RESPONSE"
				# self.sock.sendto(pickle.dumps(response), (self.send_addr, self.send_port))

				print("QBER: ", self.qber)


	




	def calc_qber(self, A, B) :
		return len(np.where(A != B))/len(A)

	def random_subset(self, A) :
		no_values = int(len(A)/10)
		return [A[0:no_values], A[(no_values+1):]]

	def sift(self, A, B) :
		return np.where(np.array(A) == np.array(B))

	def socket_listener(self) :
		print("Running listener\n")
		socket_t = Thread(target = self._socket_listener)
		socket_t.start()


	
	def cascade_confirm(self, A, B) :
		pass


	def to_basis(self, chs) :
		basis_map = {0:0, 1:0, 2:1, 3:1}
		return [basis_map[c] for c in chs]

	def to_value(self, S) :
		key_map = {0:0, 1:1, 2:0, 3:1}
		return [key_map[s] for s in S]

	def do_matching(self, alice_tags, bob_tags, radius = 1) :
		alice_matched_indices = []
		bob_matched_indices = []
		
		alice_multiple_indices = []
		bob_multiple_indices = []

		alice_tags = np.array(alice_tags)
		bob_tags = np.array(bob_tags)
		
		a_i = 0
		b_i = 0
		
		no_coin = 0
		b_start = 0
		while a_i < len(alice_tags) :
			a = alice_tags[a_i]
			b_i = b_start
			b = bob_tags[b_i]
			while b < (a + radius) and b_i < len(bob_tags):
				b = bob_tags[b_i] 
				if abs(a - b) < radius :
					if a in alice_tags[alice_matched_indices] :
						alice_multiple_indices.append(a_i)
						alice_matched_indices.remove(a_i)
					elif b in bob_tags[bob_matched_indices] :
						bob_multiple_indices.append(b_i)
						bob_matched_indices.remove(b_i)


					else :    
						alice_matched_indices.append(a_i)
						bob_matched_indices.append(b_i)
						
				b_i += 1
					
			a_i += 1
		return [alice_matched_indices, bob_matched_indices]

 

	def begin_protocol(self, tags, chs):
		if self.name == "Alice" :
			basis_bits = self.to_basis(chs)
			basis_bits = ''.join([str(b) for b in basis_bits])

			msg = QKDMsg()
			msg.msg_payload = pickle.dumps(tags)
			msg.msg_type = "MATCH_REQ"
			msg.msg_frame = 0


			self.sock.sendto(pickle.dumps(msg), (self.send_addr, self.send_port))



	def update(self, dt):
		print(self.name)
		print(self.qber)
		ttag_lock.acquire()
		if self.name == "Alice" :
			self.tags = [222, 225, 298, 353, 418, 467, 880, 886, 986, 1076, 1146, 1347, 1539, 1738, 2094, 2372, 2375, 2428, 2560, 2570, 3038, 3061, 3133, 3229, 3572, 3648, 3654, 3710, 3869, 4165, 4295, 4692, 4838, 5106, 5213, 5354, 5407, 5645, 5882, 6459, 6559, 6719, 6948, 7225, 7570, 7700, 7837, 8005, 8022, 8296, 8526, 8540, 9088, 9160, 9829, 9975, 10035, 10287, 10510, 10594, 11159, 11651, 12433, 12475, 12501, 12768, 12847, 13418, 13484, 13627, 13649, 13869, 14457, 14591, 14660, 15116, 15147, 15704, 15939, 16030, 16813, 17004, 17129, 17174, 17272, 17487, 17716, 17837, 18014, 18258, 18308, 18627, 18649, 19057, 19156, 19170, 19464, 20194, 20210, 20268, 20795, 20800, 21071, 21076, 21525, 21725, 21952, 22036, 22165, 22393, 22527, 23128, 23464, 23623, 23786, 24095, 24186, 24274, 24458, 25098, 25113, 25190, 25336, 25739, 26144, 26203, 26439, 26710, 27040, 27404, 27754, 27927, 27946, 27985, 28640, 28642, 28716, 28825, 28986, 29256, 29419, 29568, 29621, 29791, 29961, 30290, 30369, 30531, 30583, 31357, 31388, 31494, 31502, 31650, 31668, 32555, 32633, 33170, 33245, 33245, 33423, 33796, 34024, 34260, 34340, 34433, 34578, 36172, 36717, 36829, 36886, 36903, 36906, 36928, 37416, 38553, 38614, 38943, 39017, 39044, 39402, 39452, 39453, 39846, 40204, 40407, 40473, 41028, 41367, 41658, 41924, 42040, 42070, 42300, 42340, 42564, 42665, 42777, 42960, 43330, 43571, 43596, 43751, 43752, 43857, 43878, 44188, 44356, 44439, 44781, 44782, 44881, 44883, 45201, 45352, 45468, 46129, 46179, 46589, 46674, 47301, 47347, 47613, 47695, 48144, 48799, 49403, 49446, 49897, 50120, 50312, 50465, 50695, 50812, 51066, 51181, 51412, 51667, 51967, 52330, 52484, 52496, 52636, 52716, 52935, 53034, 53116, 53164, 53199, 53299, 53334, 53461, 53647, 53657, 54019, 54519, 54847, 55003, 55101, 55399, 56059, 56142, 56458, 56634, 56693, 57002, 57229, 57405, 58159, 58343, 58496, 58545, 58977, 59002, 59179, 59266, 59317, 59354, 59359, 59437, 59661, 59835, 60318, 60513, 60858, 61014, 61223, 61361, 61404, 61405, 61550, 61817, 62236, 63734, 63758, 64021, 64255, 64734, 64884, 64914, 64968, 65118, 65760, 65942, 66300, 66374, 66463, 66774, 66930, 66960, 67146, 67457, 67706, 67776, 68025, 68066, 68115, 68413, 68586, 68775, 68844, 68900, 68933, 69065, 69330, 69507, 69726, 69851, 69919, 69922, 70074, 70183, 70348, 70453, 70576, 70710, 70726, 71058, 71115, 71521, 71526, 71554, 71712, 72013, 72013, 72017, 72159, 72168, 72174, 72578, 72879, 73141, 73255, 73625, 73713, 73890, 74545, 74584, 74797, 74928, 74933, 74957, 75207, 75408, 75546, 75619, 75698, 75758, 75957, 76174, 76296, 76520, 76720, 76851, 77053, 77149, 77347, 77427, 77482, 77626, 77875, 78295, 78380, 78382, 78642, 78838, 78867, 78874, 78905, 79047, 79113, 79340, 79545, 79779, 80299, 80977, 81265, 81317, 81474, 81555, 81970, 82102, 82267, 82332, 82545, 82687, 83186, 83205, 83827, 84038, 84346, 84507, 84631, 84708, 84891, 85113, 85598, 85938, 86018, 86085, 86132, 86140, 86248, 86582, 86886, 87035, 87139, 87239, 87426, 87573, 87689, 87900, 88159, 88206, 88285, 88315, 88538, 88639, 88864, 88969, 89167, 89191, 89344, 89361, 89427, 89961, 90360, 90395, 90585, 90709, 90744, 90745, 90948, 90967, 91306, 91523, 91552, 91858, 91959, 91959, 92393, 92516, 92822, 92843, 92857, 93026, 93969, 94365, 94368, 94408, 94420, 94546, 94603, 95065, 95131, 95223, 95451, 95505, 95526, 95693, 95840, 95847, 96023, 96210, 96335, 96712, 96739, 96901, 97217, 97634, 97755, 97769, 97929, 97935, 98009, 98120, 98409, 98553, 99614, 99878] #np.copy(cur_tags)
			self.chs = [2, 2, 1, 2, 3, 1, 2, 3, 0, 3, 3, 0, 1, 3, 3, 1, 1, 0, 1, 1, 0, 3, 0, 2, 3, 1, 2, 2, 0, 3, 1, 0, 1, 3, 0, 1, 1, 1, 0, 2, 3, 3, 1, 0, 3, 0, 1, 3, 2, 2, 1, 0, 2, 3, 1, 2, 2, 1, 1, 3, 0, 3, 3, 2, 3, 1, 1, 0, 2, 1, 3, 2, 1, 1, 1, 0, 0, 1, 0, 3, 1, 2, 3, 2, 1, 2, 3, 1, 0, 3, 1, 1, 3, 3, 3, 0, 0, 0, 1, 0, 3, 0, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 1, 0, 1, 3, 3, 0, 2, 0, 2, 1, 3, 0, 3, 0, 2, 0, 3, 0, 1, 0, 0, 3, 1, 3, 2, 2, 2, 3, 3, 2, 0, 0, 2, 1, 2, 2, 2, 3, 2, 0, 0, 3, 2, 1, 3, 1, 0, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 2, 3, 3, 0, 1, 3, 3, 1, 1, 0, 0, 2, 3, 2, 3, 0, 0, 1, 0, 2, 1, 2, 1, 3, 2, 2, 2, 2, 0, 2, 0, 2, 1, 1, 1, 3, 1, 0, 3, 3, 0, 1, 1, 3, 3, 0, 3, 0, 1, 2, 1, 1, 3, 2, 0, 1, 2, 1, 2, 0, 1, 3, 0, 1, 3, 2, 0, 1, 1, 2, 0, 3, 0, 1, 0, 3, 2, 3, 1, 3, 2, 2, 3, 3, 0, 0, 0, 1, 1, 3, 2, 1, 3, 1, 1, 1, 2, 1, 1, 0, 2, 2, 2, 2, 3, 3, 3, 1, 2, 0, 2, 3, 3, 2, 1, 3, 3, 0, 2, 1, 0, 2, 2, 0, 0, 3, 0, 3, 1, 3, 1, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 1, 2, 3, 2, 2, 1, 0, 2, 3, 0, 3, 2, 3, 1, 1, 1, 0, 3, 2, 2, 2, 2, 2, 0, 2, 1, 0, 1, 0, 2, 1, 1, 1, 2, 3, 1, 0, 0, 3, 3, 0, 0, 1, 3, 3, 1, 1, 1, 3, 2, 1, 2, 1, 0, 0, 3, 0, 2, 1, 1, 2, 3, 0, 3, 1, 3, 2, 0, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 1, 0, 0, 2, 1, 2, 3, 0, 2, 3, 0, 3, 0, 1, 0, 3, 2, 1, 1, 3, 0, 3, 2, 3, 2, 3, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 3, 3, 1, 2, 3, 0, 3, 0, 1, 3, 0, 0, 3, 0, 2, 0, 3, 0, 2, 3, 1, 2, 1, 1, 1, 2, 3, 2, 2, 0, 3, 1, 2, 2, 2, 3, 2, 3, 2, 3, 0, 2, 1, 1, 2, 2, 3, 2, 3, 0, 1, 2, 3, 2, 3, 1, 2, 2, 3, 0, 0, 0, 0, 2, 0, 2, 3] # np.copy(cur_channels)
		else : 
			self.tags = [222, 225, 298, 353, 418, 467, 880, 886, 986, 1076, 1146, 1347, 1539, 1738, 2094, 2372, 2375, 2428, 2560, 2570, 3038, 3061, 3133, 3229, 3572, 3648, 3654, 3710, 3869, 4165, 4295, 4692, 4838, 5106, 5213, 5354, 5407, 5645, 5882, 6459, 6559, 6719, 6948, 7225, 7570, 7700, 7837, 8005, 8022, 8296, 8526, 8540, 9088, 9160, 9829, 9975, 10035, 10287, 10510, 10594, 11159, 11651, 12433, 12475, 12501, 12768, 12847, 13418, 13484, 13627, 13649, 13869, 14457, 14591, 14660, 15116, 15147, 15704, 15939, 16030, 16813, 17004, 17129, 17174, 17272, 17487, 17716, 17837, 18014, 18258, 18308, 18627, 18649, 19057, 19156, 19170, 19464, 20194, 20210, 20268, 20795, 20800, 21071, 21076, 21525, 21725, 21952, 22036, 22165, 22393, 22527, 23128, 23464, 23623, 23786, 24095, 24186, 24274, 24458, 25098, 25113, 25190, 25336, 25739, 26144, 26203, 26439, 26710, 27040, 27404, 27754, 27927, 27946, 27985, 28640, 28642, 28716, 28825, 28986, 29256, 29419, 29568, 29621, 29791, 29961, 30290, 30369, 30531, 30583, 31357, 31388, 31494, 31502, 31650, 31668, 32555, 32633, 33170, 33245, 33245, 33423, 33796, 34024, 34260, 34340, 34433, 34578, 36172, 36717, 36829, 36886, 36903, 36906, 36928, 37416, 38553, 38614, 38943, 39017, 39044, 39402, 39452, 39453, 39846, 40204, 40407, 40473, 41028, 41367, 41658, 41924, 42040, 42070, 42300, 42340, 42564, 42665, 42777, 42960, 43330, 43571, 43596, 43751, 43752, 43857, 43878, 44188, 44356, 44439, 44781, 44782, 44881, 44883, 45201, 45352, 45468, 46129, 46179, 46589, 46674, 47301, 47347, 47613, 47695, 48144, 48799, 49403, 49446, 49897, 50120, 50312, 50465, 50695, 50812, 51066, 51181, 51412, 51667, 51967, 52330, 52484, 52496, 52636, 52716, 52935, 53034, 53116, 53164, 53199, 53299, 53334, 53461, 53647, 53657, 54019, 54519, 54847, 55003, 55101, 55399, 56059, 56142, 56458, 56634, 56693, 57002, 57229, 57405, 58159, 58343, 58496, 58545, 58977, 59002, 59179, 59266, 59317, 59354, 59359, 59437, 59661, 59835, 60318, 60513, 60858, 61014, 61223, 61361, 61404, 61405, 61550, 61817, 62236, 63734, 63758, 64021, 64255, 64734, 64884, 64914, 64968, 65118, 65760, 65942, 66300, 66374, 66463, 66774, 66930, 66960, 67146, 67457, 67706, 67776, 68025, 68066, 68115, 68413, 68586, 68775, 68844, 68900, 68933, 69065, 69330, 69507, 69726, 69851, 69919, 69922, 70074, 70183, 70348, 70453, 70576, 70710, 70726, 71058, 71115, 71521, 71526, 71554, 71712, 72013, 72013, 72017, 72159, 72168, 72174, 72578, 72879, 73141, 73255, 73625, 73713, 73890, 74545, 74584, 74797, 74928, 74933, 74957, 75207, 75408, 75546, 75619, 75698, 75758, 75957, 76174, 76296, 76520, 76720, 76851, 77053, 77149, 77347, 77427, 77482, 77626, 77875, 78295, 78380, 78382, 78642, 78838, 78867, 78874, 78905, 79047, 79113, 79340, 79545, 79779, 80299, 80977, 81265, 81317, 81474, 81555, 81970, 82102, 82267, 82332, 82545, 82687, 83186, 83205, 83827, 84038, 84346, 84507, 84631, 84708, 84891, 85113, 85598, 85938, 86018, 86085, 86132, 86140, 86248, 86582, 86886, 87035, 87139, 87239, 87426, 87573, 87689, 87900, 88159, 88206, 88285, 88315, 88538, 88639, 88864, 88969, 89167, 89191, 89344, 89361, 89427, 89961, 90360, 90395, 90585, 90709, 90744, 90745, 90948, 90967, 91306, 91523, 91552, 91858, 91959, 91959, 92393, 92516, 92822, 92843, 92857, 93026, 93969, 94365, 94368, 94408, 94420, 94546, 94603, 95065, 95131, 95223, 95451, 95505, 95526, 95693, 95840, 95847, 96023, 96210, 96335, 96712, 96739, 96901, 97217, 97634, 97755, 97769, 97929, 97935, 98009, 98120, 98409, 98553, 99614, 99878] #np.copy(cur_tags)
			self.chs = [2, 2, 1, 2, 3, 1, 2, 3, 0, 3, 3, 0, 1, 3, 3, 1, 1, 0, 1, 1, 0, 3, 0, 2, 3, 1, 2, 2, 0, 3, 1, 0, 1, 3, 0, 1, 1, 1, 0, 2, 3, 3, 1, 0, 3, 0, 1, 3, 2, 2, 1, 0, 2, 3, 0, 2, 2, 1, 1, 3, 0, 3, 3, 2, 3, 1, 1, 0, 2, 1, 3, 2, 1, 1, 1, 0, 0, 1, 0, 3, 1, 2, 3, 2, 1, 2, 3, 1, 0, 3, 1, 1, 3, 3, 3, 0, 0, 0, 1, 0, 3, 0, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1, 2, 0, 2, 2, 0, 1, 0, 1, 3, 3, 0, 2, 0, 2, 1, 3, 0, 3, 0, 2, 0, 3, 0, 1, 0, 0, 3, 1, 3, 2, 2, 2, 3, 0, 2, 0, 0, 2, 1, 2, 2, 2, 3, 2, 0, 0, 3, 2, 1, 3, 1, 0, 0, 1, 1, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, 2, 3, 3, 0, 1, 3, 3, 1, 1, 0, 0, 2, 3, 2, 3, 0, 0, 1, 0, 2, 1, 2, 1, 3, 2, 2, 2, 2, 0, 2, 0, 2, 1, 1, 1, 3, 1, 0, 3, 3, 0, 1, 1, 3, 3, 0, 3, 0, 1, 2, 1, 1, 3, 2, 0, 1, 0, 1, 2, 0, 0, 3, 0, 1, 3, 2, 0, 1, 1, 2, 0, 3, 0, 1, 0, 3, 2, 3, 1, 3, 2, 2, 3, 3, 0, 0, 0, 1, 1, 3, 2, 1, 3, 0, 1, 1, 2, 1, 1, 0, 2, 2, 2, 2, 3, 3, 3, 1, 2, 0, 2, 3, 3, 2, 1, 3, 3, 0, 2, 1, 0, 2, 2, 0, 0, 3, 0, 3, 1, 3, 1, 0, 1, 3, 0, 3, 1, 3, 0, 3, 1, 1, 2, 3, 2, 2, 1, 0, 2, 3, 0, 3, 2, 3, 1, 1, 1, 0, 3, 2, 2, 2, 2, 2, 0, 0, 1, 0, 1, 0, 2, 1, 1, 1, 2, 3, 1, 0, 0, 3, 3, 0, 0, 1, 3, 3, 1, 1, 1, 3, 2, 1, 2, 1, 0, 0, 3, 0, 2, 1, 1, 2, 3, 0, 3, 1, 3, 2, 0, 2, 2, 3, 0, 2, 2, 2, 2, 3, 2, 1, 0, 0, 2, 1, 2, 3, 0, 2, 3, 0, 3, 0, 1, 0, 3, 2, 1, 1, 3, 0, 3, 2, 3, 2, 3, 3, 3, 1, 2, 0, 1, 2, 2, 1, 3, 3, 3, 0, 2, 3, 0, 3, 0, 1, 3, 0, 0, 3, 0, 2, 0, 3, 0, 2, 3, 1, 2, 1, 1, 1, 2, 3, 2, 2, 0, 3, 1, 2, 2, 2, 3, 2, 3, 2, 3, 0, 2, 1, 1, 2, 2, 3, 0, 0, 0, 1, 2, 0, 2, 3, 1, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 3] # np.copy(cur_channels)
		ttag_lock.release()

		self.singles_rate = len(self.tags)
		self.begin_protocol(self.tags, self.chs)

		


class HEQKDApp(App) :
	def build(self) :
		# fname = "data_input.txt"
		# poll_thread = Thread(group = None, target=poll_file, args=(fname,))
		# poll_thread.start()
		# poll_thread.join()

		self.heqkd = HEQKD()
		self.heqkd.socket_listener()
		Clock.schedule_interval(self.heqkd.update, 0.5)
		return self.heqkd


if __name__ == '__main__' :
	heqkd = HEQKDApp()
	heqkd.run()


