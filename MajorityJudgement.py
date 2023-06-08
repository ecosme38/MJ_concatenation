## Emmanuel Cosme, May 2022

import pandas as pd
import numpy as np
import copy
from IPython.display import display,Markdown
import matplotlib.pyplot as plt

class Vote():
    """
    Results of a majority Judgement vote. The votes (mentions) are read in a CSV file and saved in a Pandas DataFrame, attribute of the class, named 'df'. The majority mentions and ranks are computed at the instance creation. If argument show_results=True, the results are displayed. 
    
    Columns of the files must be 'cla', 'Id', 'Exc', 'TB', 'B', 'AB', 'Pass', 'Ins', 'AR', and 'Mmaj':
    - 'cla' stands for ranks
    - 'Id' is the identifier of the candidate. An anonymous nickname for example.
    - 'Exc', 'TB', 'B', 'AB', 'Pass', 'Ins', 'AR' are mentions. The first Nmentions are retained if Nmentions is not 7.
    - 'Mmaj' is the majority mention.
    
    All columns must be present in the file, but 'cla' and 'Mmaj' can be empty or misspecified. They are recomputed at initialization anyway.
    
    The Id column must be taken as the index list. The corresponding column number in the file should be indicated in the index_col argument (default=1).
    
    Recommendations:
    - Provide at least the file name and the number of voters in the arguments. The number of voters is necessary to convert percents in numbers of votes, if the inputs are given in percents (percent=True).
    - Once you defined an instance, you should need to use only the functions rank_candidates (to recompute MM and ranks) and display_results. All other functions are mainly for internal use, of for use in external function (df_weighted_sum below)  
    """
    
    def __init__(self, csv_file=None, sep=",", Nvoters=None, Nmentions=7, 
                 entry_type="numbers", index_col=0, show_results=True):
        
        self.mentions = ['Exc', 'TB', 'B', 'AB', 'Pass', 'Ins', 'AR'][:Nmentions]
        self.menscores = {'Exc':7, 'TB':6, 'B':5, 'AB':4, 'Pass':3, 'Ins':2, 'AR':1}
        
        self.csv_file = csv_file          # Input file name
        self.Nv = Nvoters                 # Number of voters. Needed if entries are in percents, not in numbers.
        self.formated = False             # True if the DataFrame is reformated

        ## Initialize and communicate about the type of entries
        if entry_type == "percents":
            self.percent = True            # True if the vote results are percentages
            print("Entries are in percents.")
            if self.Nv == None:
                print("If entries are in percents you must provide a number of voters with the Nvoters argument.")
        elif entry_type == "numbers":
            self.percent = False
            print("Entries are in numbers (default).")
        else:
            self.percent = False
            print("Entries are considered in numbers but the given entry_type string is not recognized.")
            print("The entry_type argument must be either percents or numbers (default).")
        
        ## Read entry csv file
        if self.csv_file != None:
            try:
                self.df = pd.read_csv(self.csv_file, sep=sep, index_col=index_col, \
                                     skipinitialspace=True)
                self.Ncand = len(self.df.index)                               # Number of candidates/identifiers
                readok = True
            except:
                self.df = []
                print('Cannot load CSV file.')
                self.Ncand = 0
                readok = False
                pass
            
            ### If file reading is successful, compute majority mentions and rank candidates
            if readok:
                self.random_reboot()
                self.reformat_dataframe()
                
                
                #### If frame in numbers, compute and check numbers of voters
                if not self.percent:
                    self.compute_number_of_voters()

                if self.percent:
                    self.percent2number()    # Convert percentages in numbers of votes
                    
                #### Compute majority mentions and rank candidates
                print("Compute majority mentions and rank candidates")
                self.rank_candidates()
                if show_results:
                    self.display_results()
                else:
                    print("To display results, type YourVoteInstance.display_results()")
       
    def random_reboot(self):
        """
        - Create columns 'cal' and 'Mmaj' if they do not exist
        - Random reboot of the majority mentions and ranks.
        """
        a = np.arange(1,self.Ncand+1)
        np.random.shuffle(a)
        self.df['cla'] = a
        self.df['Mmaj'] = np.random.choice(self.mentions,self.Ncand)
        
    def reformat_dataframe(self):
        """
        Reformat dataframe: change column mentions in Exc, TB, B, AB, Pass, Ins, AR
        """     
        ## Reformat column titles
        mention_col = self.df.columns.tolist()
        for ind, mm in enumerate(mention_col):
            if 'xc' in mm: self.df.columns.values[ind] = 'Exc'
            if 'assabl' in mm: self.df.columns.values[ind] = 'Pass'
            if 'Tr' in mm: self.df.columns.values[ind] = 'TB'
            if 'ssez' in mm: self.df.columns.values[ind] = 'AB'
            if 'rejet' in mm: self.df.columns.values[ind] = 'AR'
            if 'suffi' in mm: self.df.columns.values[ind] = 'Ins'
            if 'ien' in mm and 'Tr' not in mm: self.df.columns.values[ind] = 'B'
            
        ## Sort DataFrame columns
        self.df = self.df[self.mentions+['Mmaj','cla']]                 
        self.formated = True
    
    def compute_number_of_voters(self):
        """Compute number of voters from table of votes.
        Check that all candidates have the same number of voters and that this number is the one given by the user, if given."""
        total_voters = self.df.loc[:,self.mentions].sum(1)
        if total_voters.std() > 0:
            print("The candidates have different numbers of voters. Abort.")
            raise Exception()
        elif self.Nv != total_voters[0]:
            self.Nv = total_voters[0]
            print("The number of voters is reset to ", self.Nv)
        else:
            print("There are "+str(self.Nv)+" voters")
    
    def percent2number(self, Nv=None):
        """convert percentages in (integer) numbers of votes.
        If the number of voters is specified, the self.Nv attribute is updated."""
        if self.percent == True:
            if self.formated != True:        # Format DataFrame if not yet done
                self.reformat_dataframe()
            if Nv != None:                   # Update number of voters
                self.Nv = Nv
            if Nv == None and self.Nv == None:
                print('Oops... The number of voters must be specified to proceed')
                pass
            
            ## Convert
            tmp = self.df.loc[:,self.mentions]*self.Nv/100
            self.df.loc[:,self.mentions] = np.rint(tmp).astype(int)
            self.percent = False
            
            ## Checks
            err = np.amax(np.amax(np.abs(tmp-np.rint(tmp))))
            if err > 0.05:
                print('Watch out: conversion percents --> number of votes is uncertain:')
                print('err = ', err)
            total_voters = self.df.loc[:,self.mentions].sum(axis=1)         # Total number of votes for each identifier
            if any(total_voters != self.Nv):
                print('Oops... Numbers of voters are different for different identifiers...')
                print(total_voters)
                   
    def number2percent(self):
        """convert number of votes in percentages."""
        if self.percent == True:
            print('Votes already in percents')
        else:
            total_voters = self.df.loc[:,self.mentions].sum(axis=1)         # Total number of votes for each identifier
            ## BELOW, it can be consolidated. If non-equal, but all totnb being equal to each other, we can modify Nv...
            if all(total_voters == self.Nv):
                tmp = self.df.loc[:,self.mentions]/self.Nv*100
                self.df.loc[:,self.mentions] = np.around(tmp, decimals=1)
                self.percent = True
                
    def compute_majority_mention(self, df_in):
        """Compute majority mentions of DataFrame. Input: DataFrame to process."""
        dfin = copy.deepcopy(df_in)
        nc = len(dfin.index)                           # number of candidates
        nv = dfin.loc[:,self.mentions].iloc[0].sum()   # number of votes
        num_median = np.int64(0.5*(nv+1))              # Vote number defining the median
        cumul_vote = nv * np.ones(nc)                  # Cumulated number of votes
        for ment in dfin.loc[:,self.mentions].columns: # Loop over columns of mentions
            dfin.loc[cumul_vote>=num_median, "Mmaj"] = ment   # modify MM under appropriate condition
            cumul_vote -= dfin.loc[:,ment]             # Update by substracting the number of votes of the mention in process
        return dfin
            
    def rank_candidates(self):
        """Rank candidates using a bubble sort algorithm. Steps are:
        - random reboot of candidates ranks and MM
        - Conversion of votes from percents to numbers if necessary
        - compute majority mentions from the votes
        - bubble sort algorithm
        """
        self.random_reboot()                                 # Random reinit of MMs and ranks
        self.percent2number()                                # Convert percentages in numbers of votes
        self.df = self.compute_majority_mention(self.df)     # Compute MMs
        self.sort_dataframe()                                # sort DataFrame lines wrt to ranks
        lshift = True
        while lshift:                                        # start bubble sort algorithm
            lshift = False  
            for ic in self.df['cla'].values[:-1]:            # loop on ranks
                id1 = self.df[self.df['cla']==ic].index.tolist()[0]      # index of rank ic
                id2 = self.df[self.df['cla']==ic+1].index.tolist()[0]    # index of rank ic+1
                lshift_ = self.sort_2_candidates(id1, id2)               # Sort and shift the 2 candidates, or not
                if lshift_: lshift = True                     # If at least 1 shift, proceed bubble sort loop
        self.sort_dataframe()                                 # sort DataFrame lines wrt to ranks
            
    def sort_2_candidates(self, id1, id2):
        """Sort 2 candidates following the recursive method (https://fr.wikipedia.org/wiki/Jugement_majoritaire).
        Input: identifiers.
        Output: boolean, True is the candidates are shifted, otherwise false.
        A shift modifies the 'cla' column fo the main DataFrame."""
        df_ = self.df.loc[[id1, id2]]                     # Create DataFrame from self.df with only the two identifers
        c1, c2 = df_.loc[id1,'cla'], df_.loc[id2,'cla']   # Positions in sorting
        c10, c20 = c1, c2
        cmin, cmax = np.min([c1,c2]), np.max([c1,c2])
        m1, m2 = df_.loc[id1,'Mmaj'], df_.loc[id2,'Mmaj'] # majority mentions
        Nv_ = self.Nv  ## EC
        while self.menscores[m1] == self.menscores[m2] and Nv_>0:   # While the two mentions are similar...
            df_[m1] -= 1                                  # Remove 1 vote for the MM
            Nv_ -= 1
            df_ = self.compute_majority_mention(df_)      # Recompute MM with one less vote
            m1, m2 = df_.loc[id1,'Mmaj'], df_.loc[id2,'Mmaj']
        if self.menscores[m1] > self.menscores[m2]: c1, c2 = cmin, cmax     # rearrange if the new mentions are different
        if self.menscores[m1] < self.menscores[m2]: c1, c2 = cmax, cmin
        self.df.loc[[id1, id2], 'cla'] = [c1, c2]         # Update positions
        self.sort_dataframe()
        return c1==c20                                    # True if the candidates are shifted, False otherwise

    def barplot(self):
        """Bar plot to visualize results."""
        vsize = self.Ncand*0.5
        ax = self.df.loc[:,self.mentions].plot.barh(figsize=(14,vsize),stacked=True)
        ax.invert_yaxis()
        plt.show()
        
    def sort_dataframe(self):
        """Sort DataFrame rows in accordance with ranks."""
        self.df = self.df.sort_values(by=['cla'])
              
    def sample_N_first(self, Nc=None):
        """Sample the Nc first candidates, return result in a new DataFrame"""
        if Nc == None: Nc = self.Ncand
        self.Ncand = Nc
        self.sort_dataframe()
        return self.df.loc[self.df['cla'].isin(range(1,Nc+1))]
    
    def remove_candidates(self, ind):
        """Remove candidates from the Dataframe, return result in a new DataFrame.
        ind: list of index"""
        self.Ncand -= len(ind)
        return self.df.drop(index=ind)
    
    def display_results(self):
        """Display results: Data frame with majority mentions, ranks, and bar plot."""
        #print('                               --------------------------------------')
        print('Vote: ',self.csv_file)
        print('There are '+str(self.Ncand)+' candidates and '+str(self.Nv)+' voters.')
        display(self.df)
        print('                               --------------------------------------')
        print('Vote: ',self.csv_file)
        self.barplot()
        print('          -------------------------------------------------------------------------')
        #print('          -------------------------------------------------------------------------')
              
#### END OF Votes CLASS

def df_weighted_sum(votelist, weights, show_results=True, percents=True):
    """make weighted sum of several votes on the same candidatess.
    votelist: list of Vote instances
    weights: list of integer weights.
    percents: option for display only.
    """
    
    ## Check that the number of votes equals the number of weights
    lenv, lenw = len(votelist), len(weights)
    if lenv != lenw: print('The number of Vote instances must equal the number of weights')
    
    ## Check that the number of candidates are the same, or sample the first N, N being the smaller number among the Vote instances
    lnbcand = [ len(votelist[i].df.index) for i in range(lenv) ]
    if np.std(lnbcand) > 0:
        minnbcand = min(lnbcand)
        print('The Vote instances contain different numbers of candidates. We restrict to the smallest number: ',minnbcand)
        for vote in votelist:
            vote.df = vote.sample_N_first(Nc=minnbcand)   
    
    ## Check that candidates are the same
    ref = votelist[0].df.index.tolist()
    sameok = True
    for i in range(1,lenv):
        if set(votelist[i].df.index.tolist()) != set(ref):
            sameok = False
            print('Lists of candidates differ by:')
            print( set(ref).union(votelist[i].df.index.tolist())-set(ref).intersection(votelist[i].df.index.tolist()) )
    
    ## Create new Vote instance as weighted sum of precursors
    if sameok:
        print('Good. The candidates from the shortest list appear in the others.')
        # Create vote instance
        sum = copy.deepcopy(votelist[0])
        sum.csv_file = 'Concatenation'
        # Compute number of Voters
        _Nv, _we = np.array([toto.Nv for toto in votelist]), np.array(weights)
        sum.Nv = np.dot(_Nv, _we)
        # Compute mentions
        sum.df.loc[:,sum.mentions] = 0
        for i in range(lenv):
            votelist[i].percent2number()
            sum.df.loc[:,sum.mentions] += votelist[i].df.loc[:,sum.mentions]*weights[i]
        # Rank candidates
        sum.rank_candidates()
        if percents:
            sum.number2percent()
            print('Results are now expressed in percents.')
        if show_results:
            sum.display_results()
        return sum
    else:
        print('No concatenation nor ranking possible.')
        print('Perhaps not all candidates of the shortest list appear in the other lists...')

    