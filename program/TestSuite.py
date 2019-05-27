from ProccessTurnover import proccess_turnover 

class Stats:
	def __init__(this):
		this.fn = 0
		this.fp = 0
		this.t = 0
		this.f = 0

def assert_equals(expected, result, stat):
	if expected is True and result is False:
		stat.fn = stat.fn + 1 # false negative
	if expected is False and result is True:
		stat.fp = stat.fp + 1 # false positive
	if expected is True and result is True:
		stat.t = stat.t + 1
	if expected is False and result is False:
		stat.f = stat.f + 1

	return stat
	

stats = Stats() 
stats = assert_equals(False, proccess_turnover('vidai/1N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/2N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/3N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/4N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/5N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/6N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/7N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/8N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/9N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/10N.mp4'), stats)
stats = assert_equals(False, proccess_turnover('vidai/11N.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/1T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/2T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/3T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/4T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/5T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/6T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/7T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/8T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/9T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/10T.mp4'), stats)
stats = assert_equals(True, proccess_turnover('vidai/11T.mp4'), stats)
print("fn", stats.fn, "fp", stats.fp, "t", stats.t, "f", stats.f)