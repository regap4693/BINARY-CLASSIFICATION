from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from collections import Counter

class ImBalance():
	"""Handle ImBalance in the dataset
	"""
	def __init__(self,type):
		"""three choices for balancing - smote, oversampling, undersampling. If None base models are run i.e. with imbalanced data
		Args:
		 type: user choice of balance technique
		"""
		self.type = type
		if self.type:
			if self.type == "over":
				self.balStrategy = RandomOverSampler(random_state=42) 
			elif self.type == "smote":
				self.balStrategy = SMOTE(random_state=42) 
			else:
				self.balStrategy = RandomUnderSampler(random_state=42)
	
	def handleImbalance(self, xTrain, yTrain):
	    """Balance the Train set"""
	    if self.type:
	        print("Data is balanced using {}".format(self.type))
	        print("Shape of X and Y before balancing: ",xTrain.shape,yTrain.shape)
	        print("Class distribution: ", Counter(yTrain))
	        xTrain, yTrain = self.balStrategy.fit_resample(xTrain,yTrain)
	        print("Shape of X and Y after balancing: ",xTrain.shape,yTrain.shape)
	        print("Class distribution: ", Counter(yTrain))
	    return(xTrain, yTrain)




