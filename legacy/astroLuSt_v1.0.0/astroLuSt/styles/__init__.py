


import warnings


#filter out warnings raised by plt.style.use()
warnings.filterwarnings('ignore', message=r'.*Style includes a parameter,.+')
