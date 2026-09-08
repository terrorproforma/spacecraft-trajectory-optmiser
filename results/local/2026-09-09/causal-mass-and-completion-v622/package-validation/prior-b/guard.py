import ctypes,runpy,subprocess,sys
def blocked(*a,**k):raise RuntimeError("portable audit forbids native/process execution")
ctypes.CDLL=ctypes.PyDLL=blocked
subprocess.Popen=subprocess.run=subprocess.check_call=subprocess.check_output=blocked
runpy.run_path(sys.argv[1],run_name="__main__")
