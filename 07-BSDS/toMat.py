import scipy.io as sio

def toMat(stuff,name):
  """
  stuff:
    Structure to be sent to a .mat 
    It will be saved into a dictionary named segs
  name:
    Name of file w/o .mat 
  """

  sio.savemat(name+'.mat',{'segs':stuff})
  