import pandas as pd
import numpy as np
from tqdm import trange
import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

class GraphUtilities:
    """
    Class to process input dataset of materials' DOS and calculates its
    cumulative distribution, adjacency matrix and ForceAtlas2 positions.
    """
    def __init__(self, csv_route=None, materials_df=None):
        """
        Receives input materials, normalises them and calculates cumulative
        distribution.

        Args:
            csv_route: route for csv file with each row being different material
            materials_df: pd.DataFrame with each column being different material
        """
        if (csv_route == None) == (materials_df is None):
            raise Exception("Pass ONE argument.")

        if csv_route:
            self.materials = pd.read_csv(csv_route, header=None, index_col=0).T
        else:
            self.materials = materials_df

        self.normal_factor = self.materials.shape[0]
        self.materials_normalised = self.materials.apply(lambda col: col/col.sum())
        self.cumul_materials = self.materials_normalised.apply(lambda col: col.cumsum())
        self.cols = self.cumul_materials.columns
        self.idx = self.cols.copy()
        self.cumul_matrix = self.cumul_materials.to_numpy(dtype=float, na_value=np.nan, copy=False)

        self._diff_matrix_calculated = False
        self._adj_matrix_calculated = False
        self._fa2_node_positions_calculated = False


    def draw_sample_materials(self, number=None, random_state=6):
        """
        Randomly samples number of materials from total, and plots them.
        Args:
            number: number of materials to sample from total
            random_state: seed for sampling
        """
        if number is None:
            raise Exception("Pass the argument of the number of materials to draw.")
        elif number > self.materials.shape[1]:
            raise Exception("Number of materials to draw is larger than number of materials present.")

        sample_materials_normalised = self.materials_normalised.sample(n=number, axis='columns', random_state=random_state)
        plt.figure(figsize=(16,2*number))

        for i in range(number):
            plt.subplot(number, 1, i+1)
            plt.plot(sample_materials_normalised.iloc[:,i])
            plt.grid()
            plt.title(sample_materials_normalised.columns.values[i])


    def calculate_diff_matrix(self, method="CDF"):
        """
        Uses input method to calculate the difference matrix with no return value
        Args:
            method: method to use for calculation. Either CDF or euclidean
        """
        if (method != "CDF" and method != "euclidean"):
            raise Exception("Method must be \"CDF\" or \"euclidean\"")

        self._half_diff_matrix = np.empty((self.cumul_matrix.shape[1], self.cumul_matrix.shape[1]))
        self._half_diff_matrix.fill(0)

        if method == "CDF":
            # for i in trange(self._half_diff_matrix.shape[0]):
            #     for j in range(i+1, self._half_diff_matrix.shape[1]):
            #         self._half_diff_matrix[i][j] = sum(abs(self.cumul_matrix[:,i] - self.cumul_matrix[:,j]))
            no_calcs = self._half_diff_matrix.shape[1]-1
            for j in trange(no_calcs):
                self._half_diff_matrix[j,-no_calcs+j:] = sum(abs(
                    self.cumul_matrix[:,j+1:] - np.expand_dims(self.cumul_matrix[:,j], axis=1)
                ))

        elif method == "euclidean":
            for i in trange(self._half_diff_matrix.shape[0]):
                for j in range(i+1, self._half_diff_matrix.shape[1]):
                    self._half_diff_matrix[i][j] = np.sqrt(sum((self.cumul_matrix[:,i] - self.cumul_matrix[:,j])**2))

        self._diff_matrix = self._half_diff_matrix + self._half_diff_matrix.T
        self._diff_matrix_df = pd.DataFrame(self._diff_matrix, index=self.idx, columns=self.cols)
        self._diff_matrix_calculated = True


    def calculate_adj_matrix(self, method="one_over", a_one_over=1., a_exponential=5., b_exponential=1.):
        """
        Uses input method to calculate the adjacency matrix with no return value.
        Also calculates the difference matrix if it wasn't calculated before.
        Args:
            method: method to use for calculation. exponential (a*e^(-bx)), 
                one_minus (1 - x/516) or one_over (a/x)
            a_one_over: coefficient a for one_over
            a_exponential: coefficient b for exponential
            b_exponential: coefficient c for exponential
        """
        if (method != "exponential" and method != "one_over" and method != "one_minus"):
            raise Exception("Method must be \"exponential\", \"euclidean\" or \"one_minus\"")

        if not self._diff_matrix_calculated:
            self.calculate_diff_matrix()
        
        if method == "exponential":
            self._adj_matrix = a_exponential*np.exp(-self._diff_matrix*b_exponential/self.normal_factor)
        
        elif method == "one_minus":
            self._adj_matrix = 1 - self._diff_matrix/self.normal_factor
        
        elif method == "one_over":
            self._adj_matrix = 1/self._diff_matrix
        
        np.fill_diagonal(self._adj_matrix, 0)
        self._adj_matrix_df = pd.DataFrame(self._adj_matrix, index=self.idx, columns=self.cols)
        self._adj_matrix_df 
        self._adj_matrix_calculated = True

    
    def apply_mask_to_adj_matrix(self, method="filter_percentile_weight", weights_filter_percentile=90, keep_weights_per_material=2):
        if method not in ["filter_percentile_weight", "keep_weights_per_material"]:
            raise Exception("Method should be either 'filter_percentile_weight' or 'keep_weights_per_material'")
        if weights_filter_percentile < 0 or weights_filter_percentile > 100:
            raise Exception("Percentile filter must be between 0 and 100.")
        if type(keep_weights_per_material) != int or keep_weights_per_material < 1:
            raise Exception("keep_weights_per_material should be integer larger than 0")
        if not self._adj_matrix_calculated:
            raise Exception("Adjacency matrix hasn't been calculated.")

        if method == "filter_percentile_weight":
            adj_matrix_no_diag = self._adj_matrix[~np.eye(self._adj_matrix.shape[0],dtype=bool)].reshape(self._adj_matrix.shape[0],-1)
            top_percentile_filter = np.percentile(adj_matrix_no_diag, weights_filter_percentile)
            adj_matrix_filtered = np.copy(self._adj_matrix)
            adj_matrix_filtered[adj_matrix_filtered < top_percentile_filter] = 0
        
        elif method == "keep_weights_per_material":
            adj_matrix_filtered = np.zeros(self._adj_matrix.shape)
            for i in trange(adj_matrix_filtered.shape[1]):
                col_vals = self._adj_matrix[:i,i]
                row_vals = self._adj_matrix[i,i+1:]
                largest_n_weights = sorted(np.concatenate((col_vals, row_vals)), reverse=True)[:keep_weights_per_material]
                for idx, val in enumerate(col_vals):
                    if val in largest_n_weights:
                        adj_matrix_filtered[idx,i] = val
                for idx, val in enumerate(row_vals):
                    if val in largest_n_weights:
                        adj_matrix_filtered[i,idx+i+1] = val
                
            adj_matrix_filtered = adj_matrix_filtered + adj_matrix_filtered.T

        self._adj_matrix = adj_matrix_filtered
        self._adj_matrix_df = pd.DataFrame(adj_matrix_filtered, index=self.idx, columns=self.cols)


    def calculate_fa2_node_positions(self, number_iterations=2000):
        """
        Uses ForceAtlas2 to calculate node positions for a network of all of the
        materials in the dataset. Also calculates the difference and adjacency
        matrices if they weren't calculated before.
        """
        if not self._adj_matrix_calculated:
            self.calculate_adj_matrix()

        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=1.0,

            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,

            # Log
            verbose=True
        )
        self._fa2_positions = forceatlas2.forceatlas2(self._adj_matrix, pos=None, iterations=number_iterations)
        self._fa2_node_positions = {self._adj_matrix_df.columns.values[i]: self._fa2_positions[i] for i in range(len(self._adj_matrix_df))}


    def draw_network(self, method="matplotlib"):
        """
        Draws the full network or the network with the top percentile of the edges,
        with percentile input by user.
        Args:
            method: method used for plotting.
        """

        if not self._diff_matrix_calculated:
            self.calculate_diff_matrix()
        if not self._adj_matrix_calculated:
            self.calculate_adj_matrix()
        if not self._adj_matrix_calculated:
            self.calculate_adj_matrix()

        G = nx.from_pandas_adjacency(self._adj_matrix_df)

        if method == "matplotlib":
            if len(self._fa2_node_positions) > 100:
                plt.figure(figsize=(20,15))
                nx.draw_networkx_nodes(G, self._fa2_node_positions, node_size=25, node_color="blue", alpha=0.4)
                nx.draw_networkx_edges(G, self._fa2_node_positions, edge_color="green", alpha=0.07)
                # plt.axis('off')
            else:
                nx.draw(G, self._fa2_node_positions, with_labels=True)
                # plt.axis('off')
            plt.show()
        
        elif method == "plotly":
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = self._fa2_node_positions[edge[0]]
                x1, y1 = self._fa2_node_positions[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = self._fa2_node_positions[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    # colorscale options
                    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2))

            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_text.append(adjacencies[0] +' - # of connections: '+str(len(adjacencies[1])))

            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ),
            )
            fig.update_layout(width=1300)
            fig.update_layout(height=1000)
            fig.show()


    def get_diff_matrix(self):
        """
        Returns difference matrix, and calculates it if it wasn't calculated
        before.
        """
        if not self._diff_matrix_calculated:
            self.calculate_diff_matrix()
        
        return self._diff_matrix_df


    def get_adj_matrix(self):
        """
        Returns adjacency matrix, and calculates it if it wasn't calculated
        before.
        """
        if not self._adj_matrix_calculated:
            self.calculate_adj_matrix()
        
        return self._adj_matrix_df


    def get_fa2_node_positions(self):
        """
        Returns FA2 node positions, and calculates them if they weren't calculated
        before.
        """
        if not self._fa2_node_positions_calculated:
            self.calculate_fa2_node_positions()
        
        return self._fa2_node_positions
