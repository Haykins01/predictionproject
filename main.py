import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

#a function of an encoding system that convert alphabet to numeric based on indexing 
def label_encoder(ls):
    merr=(list(set(ls)))
    mer=[merr.index(x)+1 for x in ls]
    return (mer)



#function to removing unicode using data character ord() values as metric contain 
def remove_unicode(s):
    r=""
    for x in str(s):
        if ord(x)>47 and ord(x)<123:r+=x        
    return r

#initializing a class to controll the whole process
class Predecent:
    def __init__(xx,file,value):#the class collects two value: the file and the index of the value we want to predict
    
    # object of filename
        xx.file=file
        
        if type(value)==int:
            xx.prime_index=value
            xx.prime=None
        else:
            xx.prime=value
            xx.prime_index=None
            
        
    
        
    #function inside class to read the file using pandas
    def read_csv_file(xx,*args):        
        for read in args:
            xx.data=pd.read_csv(read)
            return (xx.data)
    
    
    #function to clean the data :filling missing data and remive Unicode using their appropriate functions     
    def clean_data(xx,*args):
        new=xx.fill_missing_data(args[0])
        xx.remove_unicode()
        
    def fill_missing_data(xx,*args):        
        missing_data_cell=xx.data.isnull().sum()
        for check in missing_data_cell:
            if check !=0:
                if len(args)<1:xx.data.fillna(data.mean(), inplace=True)                    
                else:xx.data.fillna(args[0], inplace=True)
        
        
        return (xx.data)
    
    # function to remove Unicode from Data connecting to the Unicode function above
    def remove_unicode(xx):        
        num=len(xx.data.columns.to_list())
        for m in range (num):
           xx.data.iloc[:,m]=np.array(list(map(remove_unicode,xx.data.iloc[:,m].values)))
                   
        xx.file=f"new_{xx.file}"
        xx.data.to_csv(xx.file)
        
        return xx.data

    #converting the data from categorical data to numerical data that is operatable by python     
    def data_transformation(xx):
        xx.data=pd.read_csv(xx.file)
        xx.data_s,xx.data_c=[],[]
        for x in range (len(xx.data.columns.to_list())):
            data=xx.data.iloc[:,x].values.tolist()   
            xx.data_c.append(data)         
            try:float(str(data[0]));xx.data_s.append(data)
            except Exception:xx.data_s.append(label_encoder(data))                
        return xx.data_s,xx.data_c,xx.data
        
  #genersting the mean , median and correlation between dataset      
    def generate_hyperparameter(xx,*args):        
        xx.data=args[-1]
        xx.dataset=args[-1].columns.to_list()
   
        xx.mean,xx.median=[],[]
        xx.correlation=[]
        for m,n in enumerate (xx.dataset):
            col=args[0][m]
            xx.mean.append([n,np.mean(col)])
            xx.median.append([n,np.median(col)])
            for b,d in enumerate (xx.dataset):
                col2=args[0][b]
                cor_val=np.corrcoef(col,col2)[0, 1]
                xx.correlation.append([[n,d],cor_val])
        
        print (f"Mean of individual dataset:: {xx.mean}\n\nMedian of individual dataset:: {xx.median}\n\nCorrelation between dataset:: {xx.correlation}")
        
    #plotting function using scatter plot
    def plotting (xx,*args):        
        xx.data=pd.read_csv(xx.file)
        xx.dataset=xx.data.columns.to_list()
        fig, ax = plt.subplots()
        x,y=1000,1000
        line = ax.scatter(x, y)
        plt.show(block=False)
        plt.pause(.2)
        for x,y in enumerate (args[0]):         
            b=args[0][-1]
            a=y
            print (a)
            plt.xlabel(xx.dataset[x])
            ax.set_xlim(min(a)-50,max(a)+50)
            ax.set_ylim(min(b)-50,max(b)+50)
            plt.ylabel('Pricing')
            line.set_offsets(np.c_[np.array(a),b])
            plt.show(block=False)
            plt.pause(1)
        
    
    
    #simple prediction model using the linear regression from scikit-learn module
    def prediction(xx):
       y=np.array(xx.data_s[xx.prime_index]). reshape (-1,1)
       xx.data_s.pop(xx.prime_index)
       xx.data_c.pop(xx.prime_index)
       num=len(xx.data_s)
       num2=len(xx.data_s[0])
       
       #normalizing data using the log function 
       x=np.log2(1+(np.array(xx.data_s).T)**0.5). reshape (-1,num)
       xx.data_c=np.array(xx.data_c).T. reshape (-1,num)
       
       model=LinearRegression()
       model.fit(x,y)
       
       #looping the prediction process
       while True:
           try:ind=int(input(f"\nEnter number ranging from 0 to {num2}:: "))
           except Exception: print ("\n\ninteger");ind=0
           if ind>15:ind=np.random.randint(0,15)
           res=model.predict(x[ind]. reshape (-1,num))
           print (f"\n\nFor dataset {xx.data_c[ind]} we have predicted value of {res[0][0]}\n")
           #note this prediction only works for the data ,to use new data you have to run it through some of the run function such as the data_transformation etc
           
#       function to initiate all process orderly
    def run(xx):
       data=xx.read_csv_file(xx.file)
       xx.clean_data(data)
       next=xx.data_transformation()
       xx.generate_hyperparameter(xx.data_s,xx.data_c,xx.data)   
      # xx.plotting(next[0])
       #xx.prediction()   
           
       #note you can't run the prediction and plot at the same time you have to freeze one for the other using the hash tag and completely delete the import statement for the matplotlib module 
        
        

file="data.csv"
program=Predecent(file,-12)

if __name__ == '__main__': 
    program.run()
        
