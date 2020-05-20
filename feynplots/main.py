import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from collections import Counter




class Funct:
    '''
    This takes the interaction in the graph and turns it into a function instance.
    A Function instance reads the state of the interaction and produces the function with those weights and biases
    
    If the interaction is fixed, then the function is the destandardiser.
    '''
    def __init__(self,interaction):
        self.type = interaction.type #this reads the type function either tanh, sine, gaussian or multiply
        self.state = interaction.state._to_dict() #this is the dictionary of weights
        self.sources = interaction.sources #this is the amount of inputs the function should expect.
        
    def ev(self,input0,input1=0):
        '''
        This evaluates the function at the interaction. Inputs are one-dimensional lists. This can take either one input or two.
        '''
        if self.type == 'fixed':
            '''
            The only time the function instance is used for fixed registers is when we're using the evaluating at the output register
            '''
            return self.destandardise(x=input0)
        elif self.type == 'sine':
            return self.sine(x=input0)
        elif self.type == 'gaussian':
            return self.gaussian(x=input0,y=input1)
        elif self.type == 'tanh':
            return self.tanh(x=input0,y=input1)
        elif self.type == 'multiply':
            return self.multiply(input0,input1)
        
    def destandardise(self,x):
        '''
        This is the destandardiser at the output register. The graphs output [-1,1] and then get destandardise to the range of the target feature
        '''
        feature_max = self.state['feature_max'] #max value the graph has seen from the target variable
        feature_min = self.state['feature_min'] #min value the graph has seen from the target variable
        return np.multiply((x + 1) / 2 , feature_max - feature_min ) + feature_min
               
    def sine(self,x):
        '''
        The sine function is of the form sine(k*x + x0) where k and x0 are states of the interaction
        '''
        z = np.multiply(self.state['k'],x)
        z += self.state['x0']
        return np.sin(z)
    
    def tanh(self, x,y):
        '''
        The tanh function is of the form tanh(w0*x + w1*y +bias)
        '''
        z = np.multiply(self.state['w0'],x) + np.multiply(self.state['w1'],y) + self.state['bias']
        return np.tanh(z)
    
    def gaussian(self,x,y):
        '''
        The gauss function is of the form exp( - [(x-center0)^2 / w0 + (y-center1)^2 / w1] )
        It's here where we need to know how many inputs the function has
        '''
        z = np.square(np.subtract(x,self.state['center0'])) / self.state['w0']
        if len(self.sources) == 2:
            z += np.square(np.subtract(y,self.state['center1'])) / self.state['w1'] #something wrong here when switching to one-dim gaussians
        return np.exp(-z)
    
    @staticmethod
    def multiply(x,y):
        '''
        This is the multiply function
        '''
        return np.multiply(x,y)

class Chart:
    def __init__(self,Funct):
        '''
        This chart instance will add an axis to a matplotlib figure for each interaction.
        Interactions within the graph will have [-1,1] as input ranges and an output range
        However this makes the graph a little unreadable because the actual inputs to the graph and (target variables in regression problems)
        are usually not in the range.
        The input are standardised (using MinMaxScalar) to be between [-1,1] when put in the graph
        To make the graph more readable we set the input and output ranges that we want on the axis.
        '''
        self.Funct = Funct
    
   
    def set_input_scalar_ranges(self,ranges):
        '''
        This is to set the standardisation of the inputs. Ranges takes the form of [[[x-axes input],[x-axes transform]],[[y-axes input],[y-axes transform]]]
        '''
        self.input_scalar_range0 = ranges[0]
        if len(ranges) == 2:
            self.input_scalar_range1 = ranges[1]

    def set_output_scalar_ranges(self,ranges=[[-1,1],[-1,1]]):
        '''
        This is to set the standardisation of the inputs. Ranges takes the form of [[-1,1],[output_range]]
        '''
        self.output_scalar_range = ranges

    @staticmethod    
    def minmaxscalar(x,scalar_ranges=[[-1,1],[-1,1]]):
        '''
        scalar_ranges[0] is transformed to scalar_ranges[1]
        '''
        z = np.divide(np.subtract(x,scalar_ranges[0][0]), np.subtract(scalar_ranges[0][1],scalar_ranges[0][0]))
        return np.add(np.multiply(z,np.subtract(scalar_ranges[1][1],scalar_ranges[1][0])),scalar_ranges[1][0])
    
    
    def ax_location(self,fig,gs,loc):
        '''
        In the figure, this is adding it at the gridreference
        '''
        ax = fig.add_subplot(gs[loc[0],loc[1]])
        return ax

    def plot(self,fig,gs,loc):
        '''
        This takes linearly separated points in the input range and plots the function.
        If the function takes one input then the input is on the x-axis and the output is on the y-axis
        If the function takes two inputs then the inputs are on the x and y-axes and the output are shown as contour lines in the plot
        '''
        ax = self.ax_location(fig,gs,loc)


        if len(self.Funct.sources) == 1:
            '''
            One dimensional plot
            '''
            x_axes = np.linspace(self.input_scalar_range0[0][0],self.input_scalar_range0[0][1],100) #linearly spaced points
            ax.plot(
                x_axes,Chart.minmaxscalar(
                    self.Funct.ev(
                        Chart.minmaxscalar(x_axes,scalar_ranges=self.input_scalar_range0)
                        ),
                        scalar_ranges=self.output_scalar_range)
            )   

            return fig, ax
        else:
            '''
            Contour plot
            '''
            if self.Funct.type == 'gaussian' and self.output_scalar_range[1] == [-1,1]:
                levels = [0.1,0.3,0.5,0.7,0.9]
            else:
                levels = None

            x_mesh, y_mesh = np.meshgrid(np.linspace(self.input_scalar_range0[0][0],self.input_scalar_range0[0][1],50),
                                         np.linspace(self.input_scalar_range1[0][0],self.input_scalar_range1[0][1],50))
            CS = ax.contour(
                x_mesh,y_mesh,
                Chart.minmaxscalar(
                    self.Funct.ev(
                        Chart.minmaxscalar(x_mesh,scalar_ranges=self.input_scalar_range0),
                        Chart.minmaxscalar(y_mesh,scalar_ranges=self.input_scalar_range1)
                        ),
                        scalar_ranges=self.output_scalar_range),
                        levels = levels
                        )
            ax.clabel(CS, inline=1, fontsize=10) #label for contour plot
            return fig, ax
    
    def ev(self,inputs,colour,fig,gs,loc,alpha):    
        '''
        Once we have the axis with the plot of the function, we now scatter the plot with the inputs. This is called evaluating the chart.
        Colour is a collection of numbers with the same size as an input.
        '''
        fig,ax = self.plot(fig,gs,loc)

        if len(self.Funct.sources) == 1:
            '''
            One dimensional evaluation
            '''
            cax = ax.scatter(
                x = inputs[0],
                y = Chart.minmaxscalar(
                    self.Funct.ev(
                        Chart.minmaxscalar(inputs[0],scalar_ranges=self.input_scalar_range0)
                        ),
                        scalar_ranges=self.output_scalar_range),
                        alpha = alpha, marker='o',c=colour
            )
        else:
            '''
            Two dimensional plot. Observe that there isn't any evalation of the Function here.
            '''
            cax = ax.scatter(x=inputs[0],y=inputs[1],alpha=alpha,c=colour)
            
        fig.colorbar(cax) #Adds the colour bar to the axes
        return fig,ax,cax

class GraphPlot:
    """
    A package to plot every interaction of a feyn._graph.Graph. To obtain the plot one first initalises the object with the graph.
    Then one uses graph_eval to evaluate every interaction at each datapoint. Then one uses plot to plot the figure.

    Keyword arguments:
    graph -- A feyn._graph.Graph object
    """
    def __init__(self,graph):
        self.graph = graph
        
    def graph_eval(self,data):
        """
        Evaluates each interaction in the graph at every datapoint.

        Keyword arguments:
        data -- Can be either a dict mapping register names to value arrays, or a pandas.DataFrame. Evaluates each interaction of each datapoint.
        Must include the target variable for default colour behaviour
        """ 
        self.data = data
        interactions_evaluation = np.zeros((len(self.data),len(self.graph)))
        destandardiser = Funct(self.graph[-1]) #This is the destandardiser so that the output register activation is actually the destandardised value
        for i in range(len(self.data)):
            self.graph.predict(self.data[i:i+1]) #Activation only remembers the last point that has gone through the graph. So we need to predict first...
            for j in range(len(self.graph)):
                interactions_evaluation[i][j] = self.graph[j].activation #Then find the activation value
            interactions_evaluation[i][-1] = destandardiser.ev(interactions_evaluation[i][-1]) #This is destandardising the last value in each row
        self.eval = interactions_evaluation
    
    def plot(self,colour = None, figsize=(30,20),alpha=None):
        """
        Creates a figure that contains all plots of interactions

        Keyword arguments:
        colour -- colour of scatter plot. An array of numbers that is the same length of the data that the object has been evaluated on.
        Default colour is the target variable of the data.

        figsize -- a tuple that determines the size of the figure
        alpha -- opacity of every scatter point. Takes values between 0 and 1.

        Returns:
        matplotlib.pyplot.figure -- A figure containing scatter plot of datapoints of all interactions in graph.
        """
        
        fig = plt.figure(figsize=figsize) #The figure that has all the subplots
        plt.rc('axes', labelsize=14)    # fontsize of the x and y labels

        coords, max_depth, max_height = GraphPlot._chart_locations(self.graph)
        gs = GridSpec(max_height,max_depth,figure=fig)

        output_reg = self.graph[-1] #The output register
        if colour is None:
            colour =  self.data[output_reg.name]

        interactions = [self.graph[i] for i in range(len(self.graph)) if (self.graph[i].type != 'fixed' and self.graph[i].type !='cat')] #Every interaction the requires a chart

        for interaction, coord in zip(interactions,coords):
            location = interaction._location
            chart = Chart(Funct(interaction))
            inputs = []
            input_scalar_ranges = []
            for source in interaction.sources:
                '''
                This goes through each source of the interaction and determines the inputs 
                '''
                if self.graph[source].type == 'fixed':
                    '''
                    When the source is a numerical register then the inputs comes from the data.
                    '''
                    inputs.append(self.data[self.graph[source].name]) #Input
                    feature_min = self.graph[source].state._to_dict()['feature_min']
                    feature_max = self.graph[source].state._to_dict()['feature_max']
                    input_scalar_ranges.append([[feature_min, feature_max],[-1,1]]) #The ranges of the input for the chart
                elif self.graph[source].type == 'cat':
                    '''
                    When the source is a categorical register then the inputs come from the weights at that register.
                    Fortunately this is done by the activation function at the register
                    '''
                    inputs.append([self.eval[i][source] for i in range(len(self.data))]) #Inputs
                    weight_min = min(self.graph[source].state._to_dict()['categories'], key=lambda x : x[1])[1] #minimum value
                    weight_max = max(self.graph[source].state._to_dict()['categories'], key=lambda x : x[1])[1] #maximum value
                    input_scalar_ranges.append([[weight_min,weight_max],[weight_min,weight_max]]) #Ranges of the input for the chart
                else:
                    '''
                    When the source is from an interaction that is not a register then we retrieve the inputs from the activation function
                    '''
                    inputs.append([self.eval[i][source] for i in range(len(self.data))]) #Inputs
                    input_scalar_ranges.append([[-1,1],[-1,1]]) #Range of input
                
            chart.set_input_scalar_ranges(input_scalar_ranges)
                        
            if location == (len(self.graph) - 2):
                '''
                If the interaction is the final one before the output register then we need to destandardise the output to the range of the target variable
                '''
                feature_min = output_reg.state._to_dict()['feature_min']
                feature_max = output_reg.state._to_dict()['feature_max']
                chart.set_output_scalar_ranges([[-1,1],[feature_min,feature_max]])
            else:
                '''
                Otherwise the output range is [-1,1]
                '''
                chart.set_output_scalar_ranges()

            fig, ax, cax = chart.ev(inputs,colour,fig,gs,coord,alpha) #this now plots and evaluates the chart from this interaction.
            
            #Below is labelling the axes
            if len(inputs) == 1:
                ax.set(xlabel=self.graph[interaction.sources[0]].name + ', loc: ' + str(self.graph[interaction.sources[0]]._location),
                    ylabel=interaction.name + ', loc: ' + str(interaction._location),
                    title='Interaction ' + str(interaction._location)  + ': ' + str(interaction.name))
            else:
                ax.set(xlabel=self.graph[interaction.sources[0]].name + ', loc: ' + str(self.graph[interaction.sources[0]]._location),
                    ylabel=self.graph[interaction.sources[1]].name + ', loc: ' + str(self.graph[interaction.sources[1]]._location),
                    title='Interaction ' + str(interaction._location) + ': ' + str(interaction.name))
        return fig

    def cat_plot(self, figsize=(30,20)):
        """
        A figure of bar charts of weights of categorical registers

        Keyword arguments:
        figsize -- a tuple that determines the size of the figure
        
        Returns:
        matplotlib.pyplot.figure -- A figure containing the barplots of weights of the categories for each categorical register.
        """


        cat_regs = [self.graph[i] for i in range(len(self.graph)) if self.graph[i].type == 'cat']
        fig, axs = plt.subplots(nrows = len(cat_regs), ncols = 1,figsize=figsize)

        for cat_reg in cat_regs:
            ls = cat_reg.state.categories
            ls.sort(key = lambda x : x[1])
            values = [ls[i][1] for i in range(len(ls))]
            labels = [ls[i][0] for i in range(len(ls))]
            index = cat_regs.index(cat_reg)
            axs[index].barh(range(len(ls)), values)
            axs[index].set_yticks(range(len(ls)))
            axs[index].set_yticklabels(labels)
        return fig




    @staticmethod
    def _chart_locations(graph):
        '''
        Given a graph, this provides the coordinates needed for the gridreference in a plot
        '''
        precoord1 = [graph[i].depth for i in range(len(graph)) if (graph[i].type != 'fixed' and graph[i].type !='cat')] #depth of each interaction that is not a register
        max_height = max(Counter(precoord1).values()) #maximum amount of interactions in a row in the graph
        max_depth = max(precoord1)+1 #maximum amount of interactions in a columns in the graph
        precoord0 = []
        for number in list(Counter(precoord1).values()): 
            '''
            To get the correct the coordinates the count has to restart to zero each time we move columns.
            '''
            precoord0 += [i for i in range(number)]        
        return list(zip(precoord0,precoord1)), max_depth, max_height
