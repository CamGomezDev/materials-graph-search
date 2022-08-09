import json

import umap
import numba
import numpy as np
import pandas as pd
from tqdm import trange
from fa2 import ForceAtlas2
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from materials_names import MATERIALS_NAMES

PV_MATERIALS_NAMES = {
    "mp-2534": "GaAs",
    "mp-149": "Si",
    "mp-406": "CdTe",
    "mp-20351": "InP",
    "mp-2490": "GaP",
    "mp-22736": "InCuS₂",
    "mp-1079541": "ZnCu₂SnS₄"
}

class GraphUtilities:
    """
    Class to process input dataset of materials' DOS and calculates its
    normalisation, cumulative distribution, difference and adjacency matrices,
    and calculates its node positions.
    """
    def __init__(
            self, csv_route=None, materials_df=None, band_gaps_route=None,
            es_above_hull_route=None, formation_energies_route=None
        ):
        """
        Receives input materials and routes to find the files for band gaps,
        energies above hull and formation energies.

        Args:
            csv_route: route for csv file with each row being different material
            materials_df: pd.DataFrame with each column being different material
            band_gaps_route: route for bang gaps file, each row being 
                material_id,bandgap
            es_above_hull_route: route for energies above hull file, each row
                being material_id,eabovehull
            formation_energies_route: route for formation energies file, each
                row being material_id,formation_energy
        """
        if (csv_route == None) == (materials_df is None):
            raise Exception("Pass ONE argument.")

        if csv_route:
            self.materials = pd.read_csv(csv_route, header=None, index_col=0).T
        else:
            self.materials = materials_df
        self.materials_matrix = self.materials.to_numpy()

        if band_gaps_route:
            self.bandgaps = pd.read_csv(band_gaps_route, index_col=0)
        else:
            self.bandgaps = None
        
        if es_above_hull_route:
            self.es_above_hull = pd.read_csv(es_above_hull_route, index_col=0)
        else:
            self.es_above_hull = None
        
        if formation_energies_route:
            self.formation_energies = pd.read_csv(formation_energies_route, 
                                                  index_col=0)
        else:
            self.formation_energies = None

        self._normalisation_calculated = False
        self._diff_matrix_calculated = False
        self._adj_matrix_calculated = False
        self._node_positions_calculated = False


    def draw_sample_materials(
        self, sample_number=None, sample_materials_names=None, random_state=6
    ):
        """
        Plots either a random sample or specific selected materials from the 
        dataset.
        Args:
            sample_number: number of random materials to plot from dataset
            sample_materials_names: specific materials names of materials to
                plot from dataset.
            random_state: seed for random selection of materials.
        """
        if sample_number and sample_materials_names:
            raise Exception(
                "Pass only the argument 'sample_number' or 'sample_materials'"
            )
        elif not sample_number and not sample_materials_names:
            raise Exception(
                "Pass either the argument 'sample_number' or 'sample_materials'"
            )
        elif sample_number is not None and sample_number > self.materials.shape[1]:
            raise Exception(
                "Number of materials to draw is larger than number of materials "
                "present."
            )
        
        if sample_number:
            sample_materials = self.materials.sample(n=sample_number, axis='columns', 
                                                     random_state=random_state)
            number_materials = sample_number
        elif sample_materials_names:
            real_sample_materials_names = [
                material for material in sample_materials_names 
                if material in self.materials.columns.values
            ]
            sample_materials = self.materials[real_sample_materials_names]
            number_materials = len(real_sample_materials_names)

        plt.figure(figsize=(16, 2*number_materials), facecolor="white")

        for i in range(number_materials):
            plt.subplot(number_materials, 1, i+1)
            plt.plot(sample_materials.iloc[:,i])
            plt.grid()
            plt.title(sample_materials.columns.values[i])


    def calculate_normalisation_and_cumul(self, method="same_areas"):
        """
        Calculates normalised dataset using specific method, and also calculates
        cumulative distributions of each material in the normalised dataset, with
        no return value.
        Args:
            method: method to use for calculation of normalisation. It can be
                'same_areas' (makes all materials' distributions to have the
                same area), 'standard_gaussian_scaler' (makes all materials'
                distributions to have same area using mean and standard deviation)
                or 'triple_gaussian_multiply' (divides each material's
                distribution in three, and multiplies each channel by a gaussian
                to give priority to the center)
        """
        if method not in ["same_areas", "triple_gaussian_multiply", 
                          "standard_gaussian_scaler"]:
            raise Exception("Normalisation must be 'same_areas', "
                            "'triple_gaussian_multiply' or 'standard_gaussian_scaler")

        print("Calculating normalisation and cumulative distribution...")
        
        self.normal_factor = self.materials.shape[0]

        if method == "same_areas":
            self.materials_normalised = self.materials.apply(
                lambda col: col/col.sum()
            )
            self.cumul_materials = self.materials_normalised.apply(
                lambda col: col.cumsum()
            )
            self.cumul_matrix = self.cumul_materials.to_numpy(
                dtype=float, na_value=np.nan, copy=False
            )
        
        elif method == "triple_gaussian_multiply":
            if self.normal_factor % 3 != 0:
                raise Exception("For triple channel Gaussian, the material "
                                "vector length must be divisible between 3")
            else:
                channel_length = self.normal_factor/3

                x_1 = np.arange(0, channel_length, 1)
                x_2 = np.arange(channel_length, channel_length*2, 1)
                x_3 = np.arange(channel_length*2, channel_length*3, 1)

                gauss_1 = norm.pdf(x_1, 2*channel_length/3, 40)
                gauss_2 = norm.pdf(x_2, channel_length + 2*channel_length/3, 40)
                gauss_3 = norm.pdf(x_3, channel_length*2 + 2*channel_length/3, 40)
                normal_1 = gauss_1/max(gauss_1)
                normal_2 = gauss_2/max(gauss_2)
                normal_3 = gauss_3/max(gauss_3)

                normaliser = np.concatenate((normal_1, normal_2, normal_3))

                self.materials_normalised = self.materials.apply(
                    lambda col: col*normaliser
                )
                self.cumul_materials = self.materials_normalised.apply(
                    lambda col: col.cumsum()
                )
                self.cumul_matrix = self.cumul_materials.to_numpy(
                    dtype=float, na_value=np.nan, copy=False
                )

        elif method == "standard_gaussian_scaler":
            normalised_vals = StandardScaler().fit_transform(
                self.materials.T.to_numpy()
            )
            self.materials_normalised = pd.DataFrame(
                normalised_vals.T, columns=self.materials.columns, 
                index=self.materials.index
            )
            self.cumul_materials = self.materials_normalised.apply(
                lambda col: col.cumsum()
            )
            self.cumul_matrix = self.cumul_materials.to_numpy(
                dtype=float, na_value=np.nan, copy=False
            )
        
        self._normalisation_calculated = True


    def calculate_diff_matrix(self, method="wasserstein"):
        """
        Calculates difference matrix using specific method, with no return value.
        Args:
            method: method to use for calculation. 'wasserstein' or 'euclidean'.
        """
        if (method != "wasserstein" and method != "euclidean"):
            raise Exception("Method must be 'wasserstein' or 'euclidean'")
        if method == "wasserstein" and not self._normalisation_calculated:
            raise Exception("It's necessary to calculate the normalisation before "
                            "the Wasserstein difference matrix.")
        
        print("Calculating difference matrix...")

        self._half_diff_matrix = np.empty((self.materials.shape[1], 
                                           self.materials.shape[1]))
        self._half_diff_matrix.fill(0)

        if method == "wasserstein":
            # for i in trange(self._half_diff_matrix.shape[0]):
            #     for j in range(i+1, self._half_diff_matrix.shape[1]):
            #         self._half_diff_matrix[i][j] = sum(
            #             abs(self.cumul_matrix[:,i] - self.cumul_matrix[:,j])
            #         )
            no_calcs = self._half_diff_matrix.shape[1]-1
            for j in trange(no_calcs):
                self._half_diff_matrix[j,-no_calcs+j:] = sum(abs(
                    self.cumul_matrix[:,j+1:] - \
                    np.expand_dims(self.cumul_matrix[:,j], axis=1)
                ))

        elif method == "euclidean":
            for i in trange(self._half_diff_matrix.shape[0]):
                for j in range(i+1, self._half_diff_matrix.shape[1]):
                    self._half_diff_matrix[i][j] = np.sqrt(sum(
                        (self.cumul_matrix[:,i] - self.cumul_matrix[:,j])**2
                    ))

        self._diff_matrix = self._half_diff_matrix + self._half_diff_matrix.T
        self._diff_matrix_df = pd.DataFrame(
            self._diff_matrix, index=self.cumul_materials.columns.copy(),
            columns=self.cumul_materials.columns
        )
        self._diff_matrix_calculated = True


    def calculate_adj_matrix(
        self, method="one_over", a_one_over=1., a_exponential=5., b_exponential=1.
    ):
        """
        Calculates adjacency matrix using specific method, with no return value.
        Args:
            method: method to use for calculation. exponential (a*e^(-bx)), 
                one_minus (1 - x/516) or one_over (a/x)
            a_one_over: coefficient a for one_over
            a_exponential: coefficient b for exponential
            b_exponential: coefficient c for exponential
        """
        if method not in ["exponential", "one_over", "one_minus"]:
            raise Exception(
                "Method must be 'exponential', 'euclidean' or 'one_minus'"
            )

        if not self._diff_matrix_calculated:
            raise Exception("It's necessary to calculate the difference matrix "
                            "before the adjacency one.")
        
        if method == "exponential":
            self._adj_matrix = a_exponential*\
                np.exp(-self._diff_matrix*b_exponential/self.normal_factor)
        
        elif method == "one_minus":
            self._adj_matrix = 1 - self._diff_matrix/self.normal_factor
        
        elif method == "one_over":
            self._adj_matrix = a_one_over/self._diff_matrix
        
        print("Caculating adjacency matrix...")
        
        np.fill_diagonal(self._adj_matrix, 0)
        self._adj_matrix_df = pd.DataFrame(
            self._adj_matrix, index=self.cumul_materials.columns.copy(),
            columns=self.cumul_materials.columns
        )
        self._adj_matrix_calculated = True

    
    def apply_mask_to_adj_matrix(self, 
            method="filter_percentile_weight", weights_filter_percentile=90,
            keep_weights_per_material=2
        ):
        """
        Applies a mask to adjacency matrix to remove specific elements from it,
        using specific method, with no return value.
        Args:
            method: method to apply mask. It can be 'filter_percentile_weight' 
                (keeps a top percentage of the elements, or weights, of the
                adjacency matrix) or 'keep_weights_per_material' (which keeps a
                top number of weights per material).
            weights_filter_percentile: percentage between 0 and a 100 of weights
                to remove with the 'keep_weights_per_material' method.
            keep_weights_per_material: amount of top weights per material to
                keep with 'keep_weights_per_material' method.
        """
        if not self._adj_matrix_calculated:
            raise Exception("It's necessary to calculate the adacency matrix "
                            "before applying a mask.")
        if method not in ["filter_percentile_weight", "keep_weights_per_material"]:
            raise Exception("Method should be either 'filter_percentile_weight' "
                            "or 'keep_weights_per_material'")
        if weights_filter_percentile < 0 or weights_filter_percentile > 100:
            raise Exception("Percentile filter must be between 0 and 100.")
        if type(keep_weights_per_material) != int or keep_weights_per_material < 1:
            raise Exception("keep_weights_per_material should be integer larger "
                            "than 0")
        
        print("Applying mask to adjacency matrix...")

        if method == "filter_percentile_weight":
            adj_matrix_no_diag = self._adj_matrix[
                ~np.eye(self._adj_matrix.shape[0],dtype=bool)
            ].reshape(self._adj_matrix.shape[0],-1)
            top_percentile_filter = np.percentile(adj_matrix_no_diag, 
                                                  weights_filter_percentile)
            adj_matrix_filtered = np.copy(self._adj_matrix)
            adj_matrix_filtered[adj_matrix_filtered < top_percentile_filter] = 0
        
        elif method == "keep_weights_per_material":
            adj_matrix_filtered = np.zeros(self._adj_matrix.shape)
            for i in trange(adj_matrix_filtered.shape[1]):
                col_vals = self._adj_matrix[:i,i]
                row_vals = self._adj_matrix[i,i+1:]
                largest_n_weights = sorted(
                    np.concatenate((col_vals, row_vals)), reverse=True
                )[:keep_weights_per_material]
                for idx, val in enumerate(col_vals):
                    if val in largest_n_weights:
                        adj_matrix_filtered[idx,i] = val
                for idx, val in enumerate(row_vals):
                    if val in largest_n_weights:
                        adj_matrix_filtered[i,idx+i+1] = val
                
            adj_matrix_filtered = adj_matrix_filtered + adj_matrix_filtered.T

        self._adj_matrix = adj_matrix_filtered
        self._adj_matrix_df = pd.DataFrame(
            adj_matrix_filtered, index=self.cumul_materials.columns.copy(),
            columns=self.cumul_materials.columns
        )
    

    def calculate_umap_node_positions(self, n_neighbors=15, min_dist=0.1, 
                                      distance_method="euclidean"):
        """
        Uses UMAP to calculate node positions for a network of all materials
        in the dataset, with no return value.
        Args:
            n_neighbors: number of neighbors to consider per node to its calculations.
            min_dist: minimum distance that the nodes can be to each other.
            distance_method: method to pass to UMAP for it to calculate the
                difference between nodes. It can be 'euclidean' (UMAP's default)
                or 'three_channel_wasserstein' (divides each material's
                distribution in three and calculates wasserstein per channel,
                and then sums it up).
        """
        if not self._normalisation_calculated:
            raise Exception("It's necessary to calculate the normalisation "
                            "before the difference matrix.")
        if distance_method not in ["euclidean", "three_channel_wasserstein"]:
            raise Exception("Distance method must be 'euclidean' or "
                            "'three_channel_wasserstein'")

        print("Calculating node positions...")

        if distance_method == "euclidean":
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        elif distance_method == "three_channel_wasserstein":
            reducer = umap.UMAP(metric=three_channel_wasserstein,
                                n_neighbors=n_neighbors, min_dist=min_dist)

        embeddings = reducer.fit_transform(self.materials_normalised.T.to_numpy())

        node_positions = {}
        for i, node in enumerate(self.materials.columns.values):
            node_positions[node] = embeddings[i]

        self._node_positions = node_positions
        self._node_positions_calculated = True
        

    def calculate_fa2_node_positions(self, number_iterations=2000, gravity=1.,
                                     dissuade_hubs=True):
        """
        Uses ForceAtlas2 to calculate node positions for a network of all of the
        materials in the dataset, with no return value.
        Args:
            number_iterations: number of iterations of the ForceAtlas2 algorithm.
            gravity: force of attraction between nodes of ForceAtlas2 algorithm.
            dissuade_hubs: make the ForceAtlas2 algorithm give authority to nodes.

        """
        if not self._adj_matrix_calculated:
            raise Exception("It's necessary to calculate the adjacency matrix "
                            "before calculating the node positions.")

        print("Calculating node positions...")

        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=dissuade_hubs,  # Dissuade hubs (when true,
                                                           # gives authorities to nodes)
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
            gravity=gravity,

            # Log
            verbose=True
        )
        fa2_positions = forceatlas2.forceatlas2(
            self._adj_matrix, pos=None, iterations=number_iterations
        )
        self._node_positions = {
            self._adj_matrix_df.columns.values[i]: fa2_positions[i]
            for i in range(len(self._adj_matrix_df))
        }
        self._node_positions_calculated = True


    def draw_network_matplotlib(
        self, show_bandgaps=False, show_es_above_hull=False,
        show_formation_energies=False, nodes_size=25, nodes_alpha=1, 
        show_axis=False, x_limits=None, y_limits=None, materials_emphasis=None,
        colorbar_lims=None
    ):
        """
        Draws the nodes of the network with matplotlib using previously
        calculated node positions, with no return value.
        Args:
            show_bandgaps: bool to do a band gaps color map (requires loaded 
                band gaps).
            show_es_above_hull: bool to do a energies above hull color map
                (requires loaded energies above hull).
            show_formation_energies: bool to do a formation energies color map
                (requires loaded formation energies).
            nodes_size: size of nodes to draw.
            nodes_alpha: opacity of nodes to draw.
            show_axis: show axis markers when drawing.
            x_limits: limits of x axis to draw. None or tuple (start, end).
            y_limits: litmis of y axis to draw. None or tuple (start, end).
            materials_emphasis: list of materials to emphasize.
            colorbar_lims: limits of color map for nodes.

        """
        if not self._node_positions_calculated:
            raise Exception("It's necessary to calculate the node positions "
                            "before drawing the network.")
        
        if sum([show_bandgaps, show_es_above_hull, show_formation_energies]) > 1:
            raise Exception("There can only be one color scale")
        
        plt.figure(figsize=(20,15), facecolor="white")

        materials_ordered = list(self._node_positions.keys())

        colorbar_vmin_vmax_kargs = {}
        
        if show_bandgaps or show_es_above_hull or show_formation_energies:
            if show_bandgaps:
                if type(self.bandgaps) == pd.DataFrame:
                    materials_ordered = [material for material in materials_ordered
                                         if material in self.bandgaps.index]
                    node_colors = list(self.bandgaps["band_gap"].loc[materials_ordered])
                else:
                    raise Exception("Please provide a band_gaps_route when "
                                    "instancing GraphUtilities class")
            
            if show_es_above_hull:
                if type(self.es_above_hull) == pd.DataFrame:
                    materials_ordered = [material for material in materials_ordered
                                         if material in self.es_above_hull.index]
                    node_colors = list(self.es_above_hull["ehull"].loc[materials_ordered])
                else:
                    raise Exception("Please provide a es_above_hull_route when "
                                    "instancing GraphUtilities class")
            
            if show_formation_energies:
                if type(self.formation_energies) == pd.DataFrame:
                    materials_ordered = [material for material in materials_ordered
                                         if material in self.formation_energies.index]
                    node_colors = list(self.formation_energies["formation_energy"] \
                                       .loc[materials_ordered])
                else:
                    raise Exception("Please provide a formation_energies_route "
                                    "when instancing GraphUtilities class")

            if colorbar_lims:
                colorbar_vmin_vmax_kargs = {
                    "vmin": colorbar_lims[0],
                    "vmax": colorbar_lims[1]
                }
            else:
                colorbar_vmin_vmax_kargs = {
                    "vmin": np.array(node_colors).min(),
                    "vmax": np.array(node_colors).max()
                }
        else:
            node_colors = "blue"
        
        nodes_xs = [self._node_positions[material][0] for material in materials_ordered]
        nodes_ys = [self._node_positions[material][1] for material in materials_ordered]

        plt.scatter(
            nodes_xs,
            nodes_ys,
            s=nodes_size,
            c=node_colors,
            alpha=nodes_alpha,
            **colorbar_vmin_vmax_kargs
        )

        if show_bandgaps or show_es_above_hull or show_formation_energies:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis, norm=plt.Normalize(
                    **colorbar_vmin_vmax_kargs
                )
            )
            if show_bandgaps:
                plt.colorbar(sm, label="Band gap")
            elif show_es_above_hull:
                plt.colorbar(sm, label="E above hull")
            elif show_formation_energies:
                plt.colorbar(sm, label="Formation energy")
            sm.set_array([])

        if materials_emphasis:
            emphasis_nodes_xs = [
                self._node_positions[material][0] for material in materials_ordered
                if material in materials_emphasis
            ]
            emphasis_nodes_ys = [
                self._node_positions[material][1] for material in materials_ordered
                if material in materials_emphasis
            ]
            materials_emphasis_pos = {
                material: self._node_positions[material] 
                for material in self._node_positions 
                if material in materials_emphasis
            }

            plt.scatter(
                emphasis_nodes_xs,
                emphasis_nodes_ys,
                s=nodes_size*2,
                c="red"
            )

            for material in materials_emphasis_pos:
                plt.annotate(
                    MATERIALS_NAMES.get(material, material),
                    materials_emphasis_pos[material],
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    backgroundcolor="grey",
                    fontsize=8
                )

        if x_limits:
            plt.xlim(x_limits)
        if y_limits:
            plt.ylim(y_limits)
        
        if not show_axis:
            plt.tick_params(left=False, bottom=False, labelleft=False,
                            labelbottom=False)
        
        plt.show()


    def draw_network_plotly(
            self, show_bandgaps=False, show_es_above_hull=False,
            show_formation_energies=False, nodes_size=10, materials_emphasis=[],
            nodes_alpha=1, colorbar_lims=None
        ):
        """
        Draws the nodes of the network with plotly using previously
        calculated node positions, with no return value.
        Args:
            show_bandgaps: bool to do a band gaps color map (requires loaded 
                band gaps).
            show_es_above_hull: bool to do a energies above hull color map
                (requires loaded energies above hull).
            show_formation_energies: bool to do a formation energies color map
                (requires loaded formation energies).
            nodes_size: size of nodes to draw.
            nodes_alpha: opacity of nodes to draw.
            materials_emphasis: list of materials to emphasize.
            colorbar_lims: limits of color map for nodes.
        """

        if not self._node_positions_calculated:
            raise Exception("It's necessary to calculate the node positions"
                            "before drawing the network.")
        
        materials_ordered = list(self._node_positions.keys())

        colorbar_vmin_vmax_kargs = {}

        if show_bandgaps or show_es_above_hull or show_formation_energies:
            if show_bandgaps:
                if type(self.bandgaps) == pd.DataFrame:
                    materials_ordered = [material for material in materials_ordered 
                                         if material in self.bandgaps.index]
                    node_colors = list(self.bandgaps["band_gap"]
                                       .loc[materials_ordered])
                else:
                    raise Exception("Please provide a band_gaps_route when "
                                    "instancing GraphUtilities class")
            
            if show_es_above_hull:
                if type(self.es_above_hull) == pd.DataFrame:
                    materials_ordered = [material for material in materials_ordered 
                                         if material in self.es_above_hull.index]
                    node_colors = list(self.es_above_hull["ehull"]
                                       .loc[materials_ordered])
                else:
                    raise Exception("Please provide a es_above_hull_route when "
                                    "instancing GraphUtilities class")
            
            if show_formation_energies:
                if type(self.formation_energies) == pd.DataFrame:
                    materials_ordered = [material for material in materials_ordered 
                                         if material in self.formation_energies.index]
                    node_colors = list(self.formation_energies["formation_energy"]
                                       .loc[materials_ordered])
                else:
                    raise Exception("Please provide a formation_energies_route "
                                    "when instancing GraphUtilities class")

            if colorbar_lims:
                colorbar_vmin_vmax_kargs = {
                    "cmin": colorbar_lims[0],
                    "cmax": colorbar_lims[1]
                }
            else:
                colorbar_vmin_vmax_kargs = {
                    "cmin": np.array(node_colors).min(),
                    "cmax": np.array(node_colors).max()
                }
        else:
            node_colors = ["blue"]*len(materials_ordered)

        node_x = []
        node_y = []
        node_sizes = []

        for material in materials_ordered:
            node_x.append(self._node_positions[material][0])
            node_y.append(self._node_positions[material][1])

            if material in materials_emphasis:
                node_sizes.append(nodes_size*2)
            else:
                node_sizes.append(nodes_size)

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
                size=node_sizes,
                opacity=nodes_alpha,
                colorbar=dict(
                    thickness=15,
                    # title='Node Connections',
                    xanchor='left',
                    titleside='right',
                ),
                line_width=1,
                line_color="#3d3d3d",
                **colorbar_vmin_vmax_kargs
            )
        )

        node_text = []

        for i, material in enumerate(materials_ordered):
            name = f"{MATERIALS_NAMES.get(material)} ({material})" \
                if material in MATERIALS_NAMES else material
            if show_bandgaps or show_es_above_hull or show_formation_energies:
                node_text.append(name + ' - val: ' + str(node_colors[i]))
            else:
                node_text.append(name)
            if material in materials_emphasis:
                node_colors[i] = "red"

        node_trace.marker.color = node_colors
        node_trace.text = node_text

        fig = go.Figure(
            data=[node_trace],
            layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text="Python code: <a href='https://plotly.com/ipython-not"
                         "ebooks/network-graphs/'> https://plotly.com/ipython-"
                         "notebooks/network-graphs/</a>",
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


    def get_material_top_connections(self, material, n_connections):
        """
        Returns top n_connections values in the adjacency matrix for the specified
        material.
        Args:
            material: material name to find top connections for.
            n_connections: number of connections to find in the adjacency matrix.
        Returns:
            dictionary of closest materials with their respective weights.
        """
        if not self._adj_matrix_calculated:
            raise Exception("Adjacency matrix must be calculated to find the "
                            "top connections of the material in it.")
        
        if material not in self._adj_matrix_df.columns.values:
            raise Exception("Material not in dataset used.")
        
        material_connections = self._adj_matrix_df[material]
        top_connections = material_connections.sort_values(ascending=False). \
                          iloc[:n_connections]

        return top_connections.to_dict()
    

    def get_closest_materials_to(graph_utils, orig_material, no_materials, just_names):
        """
        Get closest nodes of a material by euclidean distance in the graph with
        the calculated nodes positions.
        Args:
            orig_material: material name of the one to find closest neighbors.
            n_materials: number of closes neighbors to find for the material.
            just_names: bool to know if this function returns just a list with  
                with the names or a dictionary with the names and distances.
        Returns:
            Either a list with the names or a dictionary with the names and
                distances of the closest materials found.
        """
        material_position = graph_utils._node_positions[orig_material]
        materials_distances = []
        for material in graph_utils._node_positions:
            if material != orig_material:
                materials_distances.append((
                    material,
                    ((material_position[0] - graph_utils._node_positions[material][0])**2 + (material_position[1] - graph_utils._node_positions[material][1])**2)**0.5
                ))
        closest_materials = sorted(materials_distances, key=lambda val: val[1])[:no_materials]
        if just_names:
            return [material[0] for material in closest_materials]
        return closest_materials


    def get_diff_matrix(self):
        """
        Returns difference matrix.
        """
        if not self._diff_matrix_calculated:
            raise Exception("Difference matrix hasn't been calculated.")
        
        return self._diff_matrix_df


    def get_adj_matrix(self):
        """
        Returns adjacency matrix.
        """
        if not self._adj_matrix_calculated:
            raise Exception("Adjacency matrix hasn't been calculated.")
        
        return self._adj_matrix_df


    def get_node_positions(self):
        """
        Returns node positions.
        """
        if not self._node_positions_calculated:
            raise Exception("Node positions haven't been calculated.")
        
        return self._node_positions


    def save_diff_matrix_file(self, file_name):
        """
        Saves difference matrix to a CSV.
        """
        print("Saving difference matrix to a CSV...")
        self._diff_matrix_df.to_csv(file_name)
        print("Difference matrix saved successfully to a CSV.")


    def save_adj_matrix_file(self, file_name):
        """
        Saves adjacency matrix to a CSV.
        """
        print("Saving adjacency matrix to a CSV...")
        self._adj_matrix_df.to_csv(file_name)
        print("Adjacency matrix saved successfully to a CSV.")


    def save_node_positions_file(self, file_name):
        """
        Saves nodes positiosn to a JSON.
        """
        print("Saving node positions to a JSON...")
        with open(file_name, "w") as outfile:
            json.dump(self._node_positions, outfile)
            outfile.close()
        print("Node positions successfully saved to a JSON.")


    def load_diff_matrix_file(self, file_name):
        """
        Loads difference matrix from a CSV file.
        """
        print("Loading difference matrix from a CSV...")
        self._diff_matrix_df = pd.read_csv(file_name, index_col=0)
        self._diff_matrix = self._diff_matrix_df.to_numpy()
        self._diff_matrix_calculated = True
        print("Difference matrix successfully loaded from a CSV.")


    def load_adj_matrix_file(self, file_name):
        """
        Loads adjacency matrix from a CSV file.
        """
        print("Loading adjacency matrix from a CSV...")
        self._adj_matrix_df = pd.read_csv(file_name, index_col=0)
        self._adj_matrix = self._adj_matrix_df.to_numpy()
        self._adj_matrix_calculated = True
        print("Adjacency matrix successfully loaded from a CSV.")


    def load_node_positions_file(self, file_name):
        """
        Loads node positions from a JSON file.
        """
        print("Loading node positions from a JSON...")
        with open(file_name) as json_file:
            self._node_positions = json.load(json_file)
            json_file.close()
        self._node_positions_calculated = True
        print("Node positions successfully loaded from a JSON.")


@numba.njit()
def wasserstein(a, b):
    """
    Calculates Wasserstein distance between vectors a and b, and then sums it 
    and returns it.
    """
    a_np = np.array(a)
    b_np = np.array(b)
    a_normal = a_np/sum(a_np)
    a_normal_cumul = a_normal.cumsum()
    b_normal = b_np/sum(b_np) 
    b_normal_cumul = b_normal.cumsum()

    return sum(abs(
        a_normal_cumul - b_normal_cumul
    ))


@numba.njit()
def three_channel_wasserstein(a, b):
    """
    Divides vectors a and b vectors in three parts each, and calculates 
    Wasserstein distance between both, and the sums it and returns it.
    """
    if len(a) % 3 != 0:
        raise Exception("The length of the vectors must be divisible between 3 "
                        "to calculacute Wasserstein between three channels")

    channel_length = len(a) / 3

    distance = 0
    for i in range(3):
        a_ch_np = a[i*channel_length:(i+1)*channel_length]
        b_ch_np = b[i*channel_length:(i+1)*channel_length]
        a_ch_normal = a_ch_np/sum(a_ch_np)
        a_ch_normal_cumul = a_ch_normal.cumsum()
        b_ch_normal = b_ch_np/sum(b_ch_np)
        b_ch_normal_cumul = b_ch_normal.cumsum()
        wass_ch = np.sum(np.abs(
            a_ch_normal_cumul - b_ch_normal_cumul
        ))

        distance += wass_ch

    return distance
