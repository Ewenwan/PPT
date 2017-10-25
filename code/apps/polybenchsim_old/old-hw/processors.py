# Copyright (c) 2014. Los Alamos National Security, LLC. 

# This material was produced under U.S. Government contract DE-AC52-06NA25396
# for Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
# National Security, LLC for the U.S. Department of Energy. The U.S. Government 
# has rights to use, reproduce, and distribute this software.  

# NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, 
# EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
# If software is modified to produce derivative works, such modified software should
# be clearly marked, so as not to confuse it with the version available from LANL.

# Additionally, this library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License v 2.1 as published by the 
# Free Software Foundation. Accordingly, this library is distributed in the hope that 
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE.txt for more details.


"""
*********** Performance Prediction Toolkit PPT *********

File: processors.py
Description main library of hardware core definitions.

Comments:
 2015-11-06: included into repository, contains 
	i. 	ThreadedProcessor
	ii. KNLCore, includes Energy model
	iii.CieloCore
	iv.	MacProCore
	v. 	TTNCore (Titan, requires accelerator module)


"""

import math

#class Resource(object):
#	 """
#	 A resource is an object that is used by a process.
#	 For e.g., a computer CPU. A resource can
#	 be busy or idle; a busy resource is associated with exactly
#	 one process
#	 """
#	 def __init__(self):
#		 self.busy = False
#		 self.process = None
#		 self.process_queue = []


#class Processor(object): # define as base line, not inherited from resource
#	 """
#	 A SimX resource  that represents a computer CPU (or CPU core)
#	 """
#	 def __init__(self):
#		 super(Processor,self).__init__()
#		 self.processlist = [] # list of active processes on the core
	
class ThreadedProcessor(object): #defined as base class as other bases seem useless in SNAPSim context (1.6.15)
	"""
	A SimX resource	 that represents a computer CPU (or CPU core) with hardware threads
	"""
	def __init__(self, node):
		super(ThreadedProcessor,self).__init__()
		self.activethreads = 0 # number of active threads or processes
		self.maxthreads = 100000 # upper bound on number active threads
		self.node =	 node # needed so processor can access node memory parameters
		self.waiting_processes = [] # list of processes waiting to be executed (only non-empty if maxthreads is exceeded)




###########################################################
###########################################################
###########################################################	 
class KNLCore(ThreadedProcessor):
	"""
	A SimX resource	 that represents KnightsLanding core
	It has 2 cache levels and a vector unit
	"""
	def __init__(self, node):
		super(KNLCore, self).__init__(node)
		#
		#  PARAMETERS
		#
		self.maxthreads = 8				# upper bound on number active threads
		self.clockspeed = 1.24*10**9	# Hertz 
		self.cycles_per_CPU_ops = 5		# Number of clock cycles per CPU operation (avg)
		self.cycles_per_iALU = 2		# Number of clock cycles per integer ALU operation
		self.cycles_per_fALU = 3		# Number of clock cycles per float ALU operation
		
		self.cycles_per_vector_ops = 50 # Number of clock cycles per vector operation 
		
		self.hwthreads = 2				# number of hardware threads
		self.vector_width = 64			# width of vector unit in bytes
		self.cache_levels = 2			# number of cache levels
		self.cache_sizes = [32*10**3, 256*10**3]			  #	  list of cache sizes
		self.cache_page_sizes = [8, 1024] # list of page sizes for the different cache levels[bytes]
		self.num_registers = 64			# number of registers (single precision, holds an int) [4 bytes]
		self.register_cycles = 3		# cycles per register access
		
		self.cache_cycles = [3, 21]		# list of cache access cycles per level
		self.ram_cycles = 330			# number of cycles to load from main memory
		self.ram_page_size = 1024		# page size for main memory access [bytes]

		#
		# ENERGY PARAMETERS
		#
		self.energy_per_CPU_ops = 0             # Energy per CPU operation (avg)
		self.energy_per_iALU = 7.22099e-10		# Energy per integer ALU operation
		self.energy_per_fALU = 1.3237e-09		# Energy per float ALU operation
		self.energy_per_vector_ops = 5.11407e-17# Energy per vector operation
		self.register_energy = 8.11906e-10		# Energy per register access
		self.cache_energy = [1.23052e-08, 1.68704e-08]		# list of cache access energy per level
		self.ram_energy = 0                     # Energy to load from main memory
		self.idle_energy = 29.2759              # Energy when the processor is idle


	def time_compute(self, tasklist, statsFlag=False):
		"""
		Computes the cycles and energy that
		the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
		take 
		"""	   
		cycles = 0.0
		time = 0.0
		energy = 0.0
		stats = {}
		stats['L1_float_hits'] =0
		stats['L2_float_hits'] =0
		stats['L1_int_hits'] =0
		stats['L2_int_hits'] =0
		stats['L1_int_misses'] =0
		stats['L2_int_misses'] =0
		stats['L1_float_misses'] =0
		stats['L2_float_misses'] =0
		stats['RAM accesses'] =0
		stats['L1 cycles'] =0
		stats['L2 cycles'] =0
		stats['RAM cycles'] =0
		stats['CPU cycles'] =0
		stats['iALU cycles'] =0
		stats['fALU cycles'] =0
		stats['VECTOR ops'] =0
		stats['VECTOR cycles'] =0
		stats['internode comm time'] =0
		stats['intranode comm time'] =0
		stats['energy'] =0
		
		#print tasklist
		for item in tasklist:
			#print "Item is:", item
			#####  Memory Access #########
			if item[0] == 'MEM_ACCESS':	 
				# memory signature access, followed by 
				num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[1], item[2], item[3], item[4], item[5]
				index_loads, float_loads, init_flag = item[6], item[7], item[8]
				#print "Task list received", item
				
				#TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
				# This is a V0.1 model for realistic caching behavior
				# The arguments to MEM_ACCESS are architecture independent
				#
				# We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
				# Assume index variables are in registers as much as possible
				avg_index_loads = index_loads / num_index_vars
				num_reg_accesses = int(self.num_registers * avg_index_loads)
				nonreg_index_loads =  max(0, index_loads - num_reg_accesses) # number of index loads that are not handled by register
						
				num_vars_per_page = self.ram_page_size / avg_dist # avg number of variables per ram page 
				if init_flag: # We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
					initial_ram_pages = num_float_vars / num_vars_per_page # number of ram pages to be loaded at least once		 
					float_loads -= num_float_vars # we treat the first time that a float is loaded separately 
				else: # No init_flag false means that the float data may be already used, so no special treatment
					pass
				# Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
				L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
				L1_float_hits = int(L1_hitrate*float_loads) 
				L1_int_hits = int(L1_hitrate*nonreg_index_loads)
				
				L1_int_misses = nonreg_index_loads - L1_int_hits
				L1_float_misses = float_loads - L1_float_hits
				
				L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[1]/float(self.cache_page_sizes[1]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[1])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[1]))**2)))

				L2_float_hits = int(L2_hitrate*L1_float_misses)
				L2_int_hits = int(L2_hitrate*L1_int_misses)		 
				
				L2_int_misses = L1_int_misses - L2_int_hits
				L2_float_misses = L1_float_misses - L2_float_hits
				
				# Update the cycles number
				cycles += num_reg_accesses*self.register_cycles
				cycles += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits) # float accesses are twice as expensive
				cycles += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits) # float accesses are twice as expensive
				#cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
				cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15

				# Update the energy number
				energy += num_reg_accesses*self.register_energy
				energy += self.cache_energy[0] * (2*L1_float_hits + L1_int_hits) # float accesses are twice as expensive
				energy += self.cache_energy[1] * (2*L2_float_hits + L2_int_hits) # float accesses are twice as expensive
				#energy += self.ram_energy * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
				energy += self.ram_energy * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15

				
				stats['L1_float_hits'] += L1_float_hits
				stats['L2_float_hits'] += L2_float_hits
				stats['L1_int_hits'] += L1_int_hits
				stats['L2_int_hits'] += L2_int_hits
				stats['L1_int_misses'] += L1_int_misses
				stats['L2_int_misses'] += L2_int_misses
				stats['L1_float_misses'] += L1_float_misses
				stats['L2_float_misses'] += L2_float_misses
				stats['RAM accesses'] += ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				stats['L1 cycles'] += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits)
				stats['L2 cycles'] += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits)
				stats['RAM cycles'] += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				stats['energy'] += energy;
				
			elif item[0] == 'L1':  # L1 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[0]
				energy += num_accesses*self.cache_energy[0]
			elif item[0] == 'L2':  # L2 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[1]
				energy += num_accesses*self.cache_energy[1]
			elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:  # Higher cache access defaults to memory
				num_accesses = item[1]
				cycles += num_accesses*self.ram_cycles
				energy += num_accesses*self.ram_energy
				
			##### CPU access  ###############
			elif item[0] == 'CPU':	 # CPU ops
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_CPU_ops
				energy += num_ops*self.energy_per_CPU_ops
				stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
			elif item[0] == 'iALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_iALU
				energy += num_ops*self.energy_per_iALU
				stats['iALU cycles'] +=	 num_ops*self.cycles_per_iALU
			elif item[0] == 'fALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_fALU				  
				energy += num_ops*self.energy_per_fALU
				stats['fALU cycles'] +=	 num_ops*self.cycles_per_fALU
			elif item[0] == 'VECTOR':  # ['vector', n_ops, width ]
				num_ops = item[1]
				vec_width = item[2]
				# vec_ops is the actual number of operations necessary,	 
				# even if required width was too high.
				# the // operator is floor integer division
				vec_ops = (1+vec_width//self.vector_width)*num_ops
				cycles += vec_ops*self.cycles_per_vector_ops
				energy += vec_ops*self.energy_per_vector_ops
				stats['VECTOR ops'] +=	vec_ops
				stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
			#####  Inter-process communication #########
			elif item[0] == 'internode':	 # communication across node
				msg_size = item[1]			# number of bytes to be sent
				time += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
				stats['internode comm time'] += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
			elif item[0] == 'intranode':	 # communication across cores on same node
				num_accesses = float(item[1])/float(self.ram_page_size)		# number of bytes to be sent
											# We treat this case  as memory access
				cycles += num_accesses*self.ram_cycles
				energy += num_accesses*self.ram_energy
				stats['intranode comm time'] += num_accesses*self.ram_cycles
			#####  Memory Management #########
			elif item[0] == 'alloc':  # ['alloc', n_bytes]
				mem_size = item[1]
				if mem_size < 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				if mem_alloc_success:
					 # we will count this as a single memory access for timing purposes
					 # TODO: check validity of above assumption with Nandu
					 cycles += self.ram_cycles
					 energy += self.ram_energy
				else:
					# well, that would be a file system access then or just an exception
					# we add to time, not cycles
					time += self.node.filesystem_access_time
					#print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
					#	 " attempted allocated memory that is not available, thus causing ", \
					#	 " a filesystem action. You are in swapping mode now. "
			elif item[0] == 'unalloc':	# ['unalloc', n_bytes]
				mem_size = item[1]
				if mem_size > 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				# Unalloc just changes the memory footprint, has no impact on time	 
			################
			else:
				print 'Warning: task list item', item,' cannot be parsed, ignoring it' 
				
			time += cycles * 1/self.clockspeed * self.thread_efficiency()
			energy += cycles * 1/self.clockspeed * self.thread_efficiency() * self.idle_energy
			stats['Thread Efficiency'] = self.thread_efficiency()
		
		if statsFlag:	 
			return time, stats
		else:
			return time
		
		
	def thread_efficiency(self):
		"""
		Gives the efficiency back as a function ofthe number of active threads. Function chosen as inverse of active threads. 
		This is a cheap way of mimicing time slicing. 
		"""
		efficiency = 0.0
		#print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
		if self.activethreads <=self.hwthreads:
			efficiency = 1.0
		else:
			efficiency = float(self.hwthreads)/float(self.activethreads)
		#print "efficiency = ", efficiency
		return efficiency


	   

###########################################################
###########################################################
###########################################################	 
class CieloCore(ThreadedProcessor):
	"""
	A SimX resource	 that represents Cielo core.
	It has 3 cache levels and a vector units.
	"""
	def __init__(self, node):
		super(CieloCore, self).__init__(node)
		#
		#  PARAMETERS
		#
		self.maxthreads = 16			# upper bound on number active threads
		self.clockspeed = 2.40*10**9		# Hertz 
		self.hwthreads = 16			# number of hardware threads
		self.vector_width = 16			# width of vector unit in bytes, 128 bit?
		self.cache_levels = 3			# number of cache levels
		self.cache_sizes = [64*10**3, 512*10**3, 12*10**6]	# list of cache sizes, L1(data), L2, L3
									# is instruction cache really important to decide the performance?

		# each ops takes 1 cycle but can execute more than 1 instruction per cycle - microarchitecture
		# e.g., a lot of multiplication, alawys need to use ALU0, so not 3 muls/cycle
		# AGU/iALU/fALU 3 ops/cycle	
		# Put real cycles, we may need to introduce something like 'ILP_efficiency'
		self.cycles_per_CPU_ops = 6.4 #5.3 		# Number of clock cycles per CPU operation (avg)
		self.cycles_per_iALU = 3.9 # 2.4		# Number of clock cycles per integer ALU operation
		self.cycles_per_fALU = 6.6 # 3.5		# Number of clock cycles per float ALU operation
		self.cycles_per_vector_ops = 4.8 #3.1	# Number of clock cycles per vector operation
							# need to confirm if all SSE uses FP execution units or not

		self.cache_page_sizes = [4, 2*10**3, 10**9] 	# list of page sizes for the different cache levels [bytes]
		self.num_registers = 16				# number of registers (single precision, holds an int) [8 bytes]
		self.register_cycles = 1			# cycles per register access
								# come back later here, probably we don't need this
								# load values into registers - is this included in the actual execution (cycles per alu)
	
		self.cache_cycles = [3, 9, 30]		# list of cache access cycles per level
							# latency of L3 is variable. We just put here a guessed number for now

		self.ram_cycles = 330			# number of cycles to load from main memory
		self.ram_page_size = 1024		# page size for main memory access [bytes]


		
	def time_compute(self, tasklist, statsFlag=False):
		"""
		Computes the cycles that 
		the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
		take 
		"""	   
		cycles = 0.0
		time = 0.0
		stats = {}
		stats['L1_float_hits'] =0
		stats['L2_float_hits'] =0
		stats['L1_int_hits'] =0
		stats['L2_int_hits'] =0
		stats['L1_int_misses'] =0
		stats['L2_int_misses'] =0
		stats['L1_float_misses'] =0
		stats['L2_float_misses'] =0
		stats['RAM accesses'] =0
		stats['L1 cycles'] =0
		stats['L2 cycles'] =0
		stats['RAM cycles'] =0
		stats['CPU cycles'] =0
		stats['iALU cycles'] =0
		stats['fALU cycles'] =0
		stats['VECTOR ops'] =0
		stats['VECTOR cycles'] =0
		stats['internode comm time'] =0
		stats['intranode comm time'] =0
		
		#print tasklist
		for item in tasklist:
			#print "Item is:", item
			#####  Memory Access #########
			if item[0] == 'MEM_ACCESS':	 
				# memory signature access, followed by 
				num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[1], item[2], item[3], item[4], item[5]
				index_loads, float_loads, init_flag = item[6], item[7], item[8]
				#print "Task list received", item
				
				#TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
				# This is a V0.1 model for realistic caching behavior
				# The arguments to MEM_ACCESS are architecture independent
				#
				# We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
				# Assume index variables are in registers as much as possible
				avg_index_loads = index_loads / num_index_vars
				num_reg_accesses = int(self.num_registers * avg_index_loads)
				nonreg_index_loads =  max(0, index_loads - num_reg_accesses) # number of index loads that are not handled by register
						
				num_vars_per_page = self.ram_page_size / avg_dist # avg number of variables per ram page 
				if init_flag: # We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
					initial_ram_pages = num_float_vars / num_vars_per_page # number of ram pages to be loaded at least once		 
					float_loads -= num_float_vars # we treat the first time that a float is loaded separately 
				else: # No init_flag false means that the float data may be already used, so no special treatment
					pass
				# Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
				L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
				L1_float_hits = int(L1_hitrate*float_loads) 
				L1_int_hits = int(L1_hitrate*nonreg_index_loads)
				
				L1_int_misses = nonreg_index_loads - L1_int_hits
				L1_float_misses = float_loads - L1_float_hits
				
				L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[1]/float(self.cache_page_sizes[1]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[1])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[1]))**2)))

				L2_float_hits = int(L2_hitrate*L1_float_misses)
				L2_int_hits = int(L2_hitrate*L1_int_misses)		 
				
				L2_int_misses = L1_int_misses - L2_int_hits
				L2_float_misses = L1_float_misses - L2_float_hits
				
				L3_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[2]/float(self.cache_page_sizes[2]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[2])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[2]))**2)))

				L3_float_hits = int(L3_hitrate*L2_float_misses)
				L3_int_hits = int(L3_hitrate*L2_int_misses)		 
				
				L3_int_misses = L2_int_misses - L3_int_hits
				L3_float_misses = L2_float_misses - L3_float_hits
				
				# Update the cycles number
				cycles += num_reg_accesses*self.register_cycles
				cycles += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits) # float accesses are twice as expensive
				cycles += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits) # float accesses are twice as expensive
				cycles += self.cache_cycles[2] * (2*L3_float_hits + L3_int_hits) # float accesses are twice as expensive
				#cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
				cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15
				cycles += self.ram_cycles * ((2*L3_float_misses + L3_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15
				
				
				
				stats['L1_float_hits'] += L1_float_hits
				stats['L2_float_hits'] += L2_float_hits
				stats['L1_int_hits'] += L1_int_hits
				stats['L2_int_hits'] += L2_int_hits
				stats['L1_int_misses'] += L1_int_misses
				stats['L2_int_misses'] += L2_int_misses
				stats['L1_float_misses'] += L1_float_misses
				stats['L2_float_misses'] += L2_float_misses
				stats['RAM accesses'] += ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				stats['L1 cycles'] += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits)
				stats['L2 cycles'] += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits)
				stats['RAM cycles'] += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				
			elif item[0] == 'L1':  # L1 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[0]
			elif item[0] == 'L2':  # L2 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[1]				   
			elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:  # Higher cache access defaults to memory
				num_accesses = item[1]
				cycles += num_accesses*self.ram_cycles
				

			##### CPU access  ###############
			elif item[0] == 'CPU':	 # CPU ops
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_CPU_ops
				stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
			elif item[0] == 'iALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_iALU
				stats['iALU cycles'] +=	 num_ops*self.cycles_per_iALU
			elif item[0] == 'fALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_fALU				  
				stats['fALU cycles'] +=	 num_ops*self.cycles_per_fALU
			elif item[0] == 'VECTOR':  # ['vector', n_ops, width ]
				num_ops = item[1]
				vec_width = item[2]
				# vec_ops is the actual number of operations necessary,	 
				# even if required width was too high.
				# the // operator is floor integer division
				vec_ops = (1+vec_width//self.vector_width)*num_ops
				cycles += vec_ops*self.cycles_per_vector_ops
				stats['VECTOR ops'] +=	vec_ops
				stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
			#####  Inter-process communication #########
			elif item[0] == 'internode':	 # communication across node
				msg_size = item[1]			# number of bytes to be sent
				time += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
				stats['internode comm time'] += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
			elif item[0] == 'intranode':	 # communication across cores on same node
				num_accesses = float(item[1])/float(self.ram_page_size)		# number of bytes to be sent
											# We treat this case  as memory access
				cycles += num_accesses*self.ram_cycles
				stats['intranode comm time'] += num_accesses*self.ram_cycles
			#####  Memory Management #########
			elif item[0] == 'alloc':  # ['alloc', n_bytes]
				mem_size = item[1]
				if mem_size < 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				if mem_alloc_success:
					 # we will count this as a single memory access for timing purposes
					 # TODO: check validity of above assumption with Nandu
					 cycles += self.ram_cycles
				else:
					# well, that would be a file system access then or just an exception
					# we add to time, not cycles
					time += self.node.filesystem_access_time
					#print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
					#	 " attempted allocated memory that is not available, thus causing ", \
					#	 " a filesystem action. You are in swapping mode now. "
			elif item[0] == 'unalloc':	# ['unalloc', n_bytes]
				mem_size = item[1]
				if mem_size > 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				# Unalloc just changes the memory footprint, has no impact on time	 
			################
			else:
				print 'Warning: task list item', item,' cannot be parsed, ignoring it' 
				
			time += cycles * 1/self.clockspeed * self.thread_efficiency()
			stats['Thread Efficiency'] = self.thread_efficiency()
		
		if statsFlag:	 
			return time, stats
		else:
			return time
		
		
	def thread_efficiency(self):
		"""
		Gives the efficiency back as a function ofthe number of active threads. Function chosen as inverse of active threads. 
		This is a cheap way of mimicing time slicing. 
		"""
		efficiency = 0.0
		#print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
		if self.activethreads <=self.hwthreads:
			efficiency = 1.0
		else:
			efficiency = float(self.hwthreads)/float(self.activethreads)
		#print "efficiency = ", efficiency
		return efficiency
	   



###########################################################
###########################################################
###########################################################	 
class MacProCore(ThreadedProcessor):
	"""
	A SimX resource	 that represents MacPro core.
	It has 3 cache levels and a vector units.
	"""
	def __init__(self, node):
		super(MacProCore, self).__init__(node)
		#
		#  PARAMETERS
		#
		self.maxthreads = 24			# upper bound on number active threads
		self.clockspeed = 2.7*10**9		# Hertz 
		self.hwthreads = 12			# number of hardware threads, EJ: might be 24, but I am almost sure 12, need to confirm
		self.vector_width = 32			# width of vector unit in bytes, 256 bit
		self.cache_levels = 3			# number of cache levels
		self.cache_sizes = [12*10**3, 256*10**3, 30*10**6]	# list of cache sizes, L1(data), L2, L3
									# is instruction cache really important to decide the performance?

		# each ops takes 1 cycle but can execute more than 1 instruction per cycle - microarchitecture
		# e.g., a lot of multiplication, alawys need to use ALU0, so not 3 muls/cycle
		# AGU/iALU/fALU 3 ops/cycle	
		# Put real cycles, we may need to introduce something like 'ILP_efficiency'
		self.cycles_per_CPU_ops = 1		# Number of clock cycles per CPU operation (avg)
		self.cycles_per_iALU = 1		# Number of clock cycles per integer ALU operation
		self.cycles_per_fALU = 1		# Number of clock cycles per float ALU operation
		self.cycles_per_vector_ops = 1		# Number of clock cycles per vector operation
							# need to confirm if all SSE uses FP execution units or not

		self.cache_page_sizes = [4, 2*10**3, 10**9] 	# list of page sizes for the different cache levels [bytes]
		self.num_registers = 16				# number of registers (single precision, holds an int) [8 bytes]
		self.register_cycles = 1			# cycles per register access
								# come back later here, probably we don't need this
								# load values into registers - is this included in the actual execution (cycles per alu)
	
		self.cache_cycles = [3, 9, 30]		# list of cache access cycles per level
							# latency of L3 is variable. We just put here a guessed number for now

		self.ram_cycles = 330			# number of cycles to load from main memory
		self.ram_page_size = 1024		# page size for main memory access [bytes]


		
	def time_compute(self, tasklist, statsFlag=False):
		"""
		Computes the cycles that 
		the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
		take 
		"""	   
		cycles = 0.0
		time = 0.0
		stats = {}
		stats['L1_float_hits'] =0
		stats['L2_float_hits'] =0
		stats['L1_int_hits'] =0
		stats['L2_int_hits'] =0
		stats['L1_int_misses'] =0
		stats['L2_int_misses'] =0
		stats['L1_float_misses'] =0
		stats['L2_float_misses'] =0
		stats['RAM accesses'] =0
		stats['L1 cycles'] =0
		stats['L2 cycles'] =0
		stats['RAM cycles'] =0
		stats['CPU cycles'] =0
		stats['iALU cycles'] =0
		stats['fALU cycles'] =0
		stats['VECTOR ops'] =0
		stats['VECTOR cycles'] =0
		stats['internode comm time'] =0
		stats['intranode comm time'] =0
		
		#print tasklist
		for item in tasklist:
			#print "Item is:", item
			#####  Memory Access #########
			if item[0] == 'MEM_ACCESS':	 
				# memory signature access, followed by 
				num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[1], item[2], item[3], item[4], item[5]
				index_loads, float_loads, init_flag = item[6], item[7], item[8]
				#print "Task list received", item
				
				#TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
				# This is a V0.1 model for realistic caching behavior
				# The arguments to MEM_ACCESS are architecture independent
				#
				# We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
				# Assume index variables are in registers as much as possible
				avg_index_loads = index_loads / num_index_vars
				num_reg_accesses = int(self.num_registers * avg_index_loads)
				nonreg_index_loads =  max(0, index_loads - num_reg_accesses) # number of index loads that are not handled by register
						
				num_vars_per_page = self.ram_page_size / avg_dist # avg number of variables per ram page 
				if init_flag: # We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
					initial_ram_pages = num_float_vars / num_vars_per_page # number of ram pages to be loaded at least once		 
					float_loads -= num_float_vars # we treat the first time that a float is loaded separately 
				else: # No init_flag false means that the float data may be already used, so no special treatment
					pass
				# Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
				L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
				L1_float_hits = int(L1_hitrate*float_loads) 
				L1_int_hits = int(L1_hitrate*nonreg_index_loads)
				
				L1_int_misses = nonreg_index_loads - L1_int_hits
				L1_float_misses = float_loads - L1_float_hits
				
				L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[1]/float(self.cache_page_sizes[1]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[1])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[1]))**2)))

				L2_float_hits = int(L2_hitrate*L1_float_misses)
				L2_int_hits = int(L2_hitrate*L1_int_misses)		 
				
				L2_int_misses = L1_int_misses - L2_int_hits
				L2_float_misses = L1_float_misses - L2_float_hits
				
				L3_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[2]/float(self.cache_page_sizes[2]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[2])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[2]))**2)))

				L3_float_hits = int(L3_hitrate*L2_float_misses)
				L3_int_hits = int(L3_hitrate*L2_int_misses)		 
				
				L3_int_misses = L2_int_misses - L3_int_hits
				L3_float_misses = L2_float_misses - L3_float_hits
				
				# Update the cycles number
				cycles += num_reg_accesses*self.register_cycles
				cycles += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits) # float accesses are twice as expensive
				cycles += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits) # float accesses are twice as expensive
				cycles += self.cache_cycles[2] * (2*L3_float_hits + L3_int_hits) # float accesses are twice as expensive
				#cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
				cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15
				cycles += self.ram_cycles * ((2*L3_float_misses + L3_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15
				
				
				
				stats['L1_float_hits'] += L1_float_hits
				stats['L2_float_hits'] += L2_float_hits
				stats['L1_int_hits'] += L1_int_hits
				stats['L2_int_hits'] += L2_int_hits
				stats['L1_int_misses'] += L1_int_misses
				stats['L2_int_misses'] += L2_int_misses
				stats['L1_float_misses'] += L1_float_misses
				stats['L2_float_misses'] += L2_float_misses
				stats['RAM accesses'] += ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				stats['L1 cycles'] += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits)
				stats['L2 cycles'] += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits)
				stats['RAM cycles'] += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				
			elif item[0] == 'L1':  # L1 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[0]
			elif item[0] == 'L2':  # L2 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[1]				   
			elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:  # Higher cache access defaults to memory
				num_accesses = item[1]
				cycles += num_accesses*self.ram_cycles
				

			##### CPU access  ###############
			elif item[0] == 'CPU':	 # CPU ops
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_CPU_ops
				stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
			elif item[0] == 'iALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_iALU
				stats['iALU cycles'] +=	 num_ops*self.cycles_per_iALU
			elif item[0] == 'fALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_fALU				  
				stats['fALU cycles'] +=	 num_ops*self.cycles_per_fALU
			elif item[0] == 'VECTOR':  # ['vector', n_ops, width ]
				num_ops = item[1]
				vec_width = item[2]
				# vec_ops is the actual number of operations necessary,	 
				# even if required width was too high.
				# the // operator is floor integer division
				vec_ops = (1+vec_width//self.vector_width)*num_ops
				cycles += vec_ops*self.cycles_per_vector_ops
				stats['VECTOR ops'] +=	vec_ops
				stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
			#####  Inter-process communication #########
			elif item[0] == 'internode':	 # communication across node
				msg_size = item[1]			# number of bytes to be sent
				time += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
				stats['internode comm time'] += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
			elif item[0] == 'intranode':	 # communication across cores on same node
				num_accesses = float(item[1])/float(self.ram_page_size)		# number of bytes to be sent
											# We treat this case  as memory access
				cycles += num_accesses*self.ram_cycles
				stats['intranode comm time'] += num_accesses*self.ram_cycles
			#####  Memory Management #########
			elif item[0] == 'alloc':  # ['alloc', n_bytes]
				mem_size = item[1]
				if mem_size < 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				if mem_alloc_success:
					 # we will count this as a single memory access for timing purposes
					 # TODO: check validity of above assumption with Nandu
					 cycles += self.ram_cycles
				else:
					# well, that would be a file system access then or just an exception
					# we add to time, not cycles
					time += self.node.filesystem_access_time
					#print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
					#	 " attempted allocated memory that is not available, thus causing ", \
					#	 " a filesystem action. You are in swapping mode now. "
			elif item[0] == 'unalloc':	# ['unalloc', n_bytes]
				mem_size = item[1]
				if mem_size > 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				# Unalloc just changes the memory footprint, has no impact on time	 
			################
			else:
				print 'Warning: task list item', item,' cannot be parsed, ignoring it' 
				
			time += cycles * 1/self.clockspeed * self.thread_efficiency()
			stats['Thread Efficiency'] = self.thread_efficiency()
		
		if statsFlag:	 
			return time, stats
		else:
			return time
		
		
	def thread_efficiency(self):
		"""
		Gives the efficiency back as a function ofthe number of active threads. Function chosen as inverse of active threads. 
		This is a cheap way of mimicing time slicing. 
		"""
		efficiency = 0.0
		#print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
		if self.activethreads <=self.hwthreads:
			efficiency = 1.0
		else:
			efficiency = float(self.hwthreads)/float(self.activethreads)
		#print "efficiency = ", efficiency
		return efficiency
	   

###########################################################
###########################################################
###########################################################	 
class TTNCore(ThreadedProcessor):
	"""
	A SimX resource	 that represents Titan core
	It has 2 cache levels and a vector unit
	"""
	def __init__(self, node):
		super(TTNCore, self).__init__(node)
		#
		#  PARAMETERS
		#
		self.maxthreads = 16			# upper bound on number active threads
		self.hwthreads = 16				# number of hardware threads
		self.clockspeed = 2.2*10**9		# Hertz 
		self.cycles_per_CPU_ops = 5		# Number of clock cycles per CPU operation (avg)
		self.cycles_per_iALU = .5		# Number of clock cycles per integer ALU operation
		self.cycles_per_fALU = .5		# Number of clock cycles per float ALU operation
		
		self.cycles_per_vector_ops = 5 	# Number of clock cycles per vector operation 
		
		self.vector_width = 64			# width of vector unit in bytes
		self.cache_levels = 3			# number of cache levels
		self.cache_sizes = [48*10**3, 1*10**6, 16*10**6]			  #	  list of cache sizes
		self.cache_page_sizes = [16, 16, ] 	# list of page sizes for the different cache levels[bytes]
		self.num_registers = 64			# number of registers (single precision, holds an int) [4 bytes]
		self.register_cycles = 3		# cycles per register access
		
		self.cache_cycles = [3.5, 17.8, 53.9]	# list of cache access cycles per level
		self.ram_cycles = 150			# number of cycles to load from main memory
		self.ram_page_size = 1024		# page size for main memory access [bytes]
		self.end_device_comps = []		# Contains the number of CPU cycles after which each device will finish
										# computing or 0 if no computation is being run the device
		for i in range(self.node.num_accelerators):
			self.end_device_comps.append(0)
		


		
	def time_compute(self, tasklist, start, statsFlag=False):
		"""
		Computes the cycles that 
		the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
		take 
		"""	   
		cycles = 0.0
		time = 0.0
		stats = {}
		stats['L1_float_hits'] =0
		stats['L2_float_hits'] =0
		stats['L1_int_hits'] =0
		stats['L2_int_hits'] =0
		stats['L1_int_misses'] =0
		stats['L2_int_misses'] =0
		stats['L1_float_misses'] =0
		stats['L2_float_misses'] =0
		stats['RAM accesses'] =0
		stats['L1 cycles'] =0
		stats['L2 cycles'] =0
		stats['RAM cycles'] =0
		stats['CPU cycles'] =0
		stats['iALU cycles'] =0
		stats['fALU cycles'] =0
		stats['VECTOR ops'] =0
		stats['VECTOR cycles'] =0
		stats['internode comm time'] =0
		stats['intranode comm time'] =0
		
		#print tasklist
		for item in tasklist:
			#print "Item is:", item
			#####  Memory Access #########
			if item[0] == 'MEM_ACCESS':	 
				# memory signature access, followed by 
				num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[1], item[2], item[3], item[4], item[5]
				index_loads, float_loads, init_flag = item[6], item[7], item[8]
				#print "Task list received", item
				
				#TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
				# This is a V0.1 model for realistic caching behavior
				# The arguments to MEM_ACCESS are architecture independent
				#
				# We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
				# Assume index variables are in registers as much as possible
				avg_index_loads = index_loads / num_index_vars
				num_reg_accesses = int(self.num_registers * avg_index_loads)
				nonreg_index_loads =  max(0, index_loads - num_reg_accesses) # number of index loads that are not handled by register
						
				num_vars_per_page = self.ram_page_size / avg_dist # avg number of variables per ram page 
				if init_flag: # We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
					initial_ram_pages = num_float_vars / num_vars_per_page # number of ram pages to be loaded at least once		 
					float_loads -= num_float_vars # we treat the first time that a float is loaded separately 
				else: # No init_flag false means that the float data may be already used, so no special treatment
					pass
				# Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
				L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
				L1_float_hits = int(L1_hitrate*float_loads) 
				L1_int_hits = int(L1_hitrate*nonreg_index_loads)
				
				L1_int_misses = nonreg_index_loads - L1_int_hits
				L1_float_misses = float_loads - L1_float_hits
				
				L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[1]/float(self.cache_page_sizes[1]) - \
					(avg_reuse_dist/float(self.cache_page_sizes[1])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[1]))**2)))

				L2_float_hits = int(L2_hitrate*L1_float_misses)
				L2_int_hits = int(L2_hitrate*L1_int_misses)		 
				
				L2_int_misses = L1_int_misses - L2_int_hits
				L2_float_misses = L1_float_misses - L2_float_hits
				
				# Update the cycles number
				cycles += num_reg_accesses*self.register_cycles
				cycles += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits) # float accesses are twice as expensive
				cycles += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits) # float accesses are twice as expensive
				#cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
				cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15
				
				
				stats['L1_float_hits'] += L1_float_hits
				stats['L2_float_hits'] += L2_float_hits
				stats['L1_int_hits'] += L1_int_hits
				stats['L2_int_hits'] += L2_int_hits
				stats['L1_int_misses'] += L1_int_misses
				stats['L2_int_misses'] += L2_int_misses
				stats['L1_float_misses'] += L1_float_misses
				stats['L2_float_misses'] += L2_float_misses
				stats['RAM accesses'] += ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				stats['L1 cycles'] += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits)
				stats['L2 cycles'] += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits)
				stats['RAM cycles'] += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
				
			elif item[0] == 'L1':  # L1 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[0]
			elif item[0] == 'L2':  # L2 accesses, followed by number
				num_accesses = item[1]
				cycles += num_accesses*self.cache_cycles[1]				   
			elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:  # Higher cache access defaults to memory
				num_accesses = item[1]
				cycles += num_accesses*self.ram_cycles
				
			##### CPU access  ###############
			elif item[0] == 'CPU':	 # CPU ops
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_CPU_ops
				stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
			elif item[0] == 'iALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_iALU
				stats['iALU cycles'] +=	 num_ops*self.cycles_per_iALU
			elif item[0] == 'fALU':	  # Integer additions
				num_ops = item[1]
				cycles += num_ops*self.cycles_per_fALU				  
				stats['fALU cycles'] +=	 num_ops*self.cycles_per_fALU
			elif item[0] == 'VECTOR':  # ['vector', n_ops, width ]
				num_ops = item[1]
				vec_width = item[2]
				# vec_ops is the actual number of operations necessary,	 
				# even if required width was too high.
				# the // operator is floor integer division
				vec_ops = (1+vec_width//self.vector_width)*num_ops
				cycles += vec_ops*self.cycles_per_vector_ops
				stats['VECTOR ops'] +=	vec_ops
				stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
			#####  Inter-process communication #########
			elif item[0] == 'internode':	 # communication across node
				msg_size = item[1]			# number of bytes to be sent
				time += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
				stats['internode comm time'] += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
			elif item[0] == 'intranode':	# communication across cores on same node
				num_accesses = item[1]		# number of bytes to be sent
											# We treat this case  as memory access
				cycles += num_accesses*self.ram_cycles
				stats['intranode comm time'] += num_accesses*self.ram_cycles
			#####  Memory Management #########
			elif item[0] == 'alloc':  # ['alloc', n_bytes]
				mem_size = item[1]
				if mem_size < 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				if mem_alloc_success:
					 # we will count this as a single memory access for timing purposes
					 # TODO: check validity of above assumption with Nandu
					 cycles += self.ram_cycles
				else:
					# well, that would be a file system access then or just an exception
					# we add to time, not cycles
					time += self.node.filesystem_access_time
					#print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
					#	 " attempted allocated memory that is not available, thus causing ", \
					#	 " a filesystem action. You are in swapping mode now. "
			elif item[0] == 'unalloc':	# ['unalloc', n_bytes]
				mem_size = item[1]
				if mem_size > 0:
					mem_size = - mem_size
				mem_alloc_success = self.node.mem_alloc(mem_size)
				# Unalloc just changes the memory footprint, has no impact on time	 
				
			elif item[0] == 'DEVICE_ALLOC':
				if item[1] >= self.node.num_accelerators:
					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
				else:
					self.node.accelerators[item[1]].mem_alloc(item[2])
					
			elif item[0] == 'DEVICE_TRANSFER':
				if item[1] >= self.node.num_accelerators:
					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
				else:
					cycles += self.node.accelerators[item[1]].transfer(item[2])*self.clockspeed
					
			elif item[0] == 'KERNEL_CALL':
				if item[1] >= self.node.num_accelerators:
					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
				else:
					# Compute the number of *CPU* cycles during which the device will be busy
					# This number will be added to the total number of cycles when synchronizing with the GPU
					# under the condition that it is higher than the current number of cycles (concurrency between CPU and device)
					self.end_device_comps[item[1]] = cycles+self.node.accelerators[item[1]].kernel_call(item[2], item[3], item[4], item[5], start+cycles*1/self.clockspeed)*self.clockspeed
			elif item[0] == 'DEVICE_SYNC':
				if item[1] >= self.node.num_accelerators:
					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
				else:
					cycles = max(self.end_device_comps[item[1]], cycles)
					
				
			################
			else:
				print 'Warning: task list item', item,' cannot be parsed, ignoring it' 
					
		time += cycles * 1/self.clockspeed * self.thread_efficiency()
		stats['Thread Efficiency'] = self.thread_efficiency()
		
		if statsFlag:	 
			return time, stats
		else:
			return time
		
		
	def thread_efficiency(self):
		"""
		Gives the efficiency back as a function of the number of active threads. Function chosen as inverse of active threads. 
		This is a cheap way of mimicing time slicing. 
		"""
		efficiency = 0.0
		#print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
		if self.activethreads <=self.hwthreads:
			efficiency = 1.0
		else:
			efficiency = float(self.hwthreads)/float(self.activethreads)
		#print "efficiency = ", efficiency
		return efficiency
