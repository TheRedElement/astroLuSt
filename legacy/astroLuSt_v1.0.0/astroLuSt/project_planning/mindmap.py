

#%%imports
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import re

#%%definitions
class MindMap:
    """
        - class to create a very primitive mindmap using python code

        Attributes
        ----------
            - node_contents
                - list of strings
                - text corresponding to the node
            - node_levels
                - list of int
                - levels corresponding to the node
                - a root is defined as node_level == 0
            - hide_branch
                - list of bools
                - whether to hide all the following nodes
                - NOT IMPLEMENTED YET
            - edge_froms
                - list of strings
                - origins of all edges between nodes
                - the entries are entries of 'node_contents'
            - edge_tos
                - list of strings
                - ends of all edges between nodes
                - the entries are entries of 'node_contents'
            - edge_weights
                - list of float
                - weight for each edge
                - determines size of edge in graph
            - edge_types
                - list of str
                - which line-style to use when drawing the edge
                
        Methods
        -------
            - get_nodes_edges
                - function to get dicts of nodes and edges
            - check_keys
                - function to check if correct keys are provided to relationships and carac
            - generate_node_colors
                - function to generate colors for plotting the nodes
            - get_fibonacci_disc
                - function to generate a fibonacci disc
                - used to generate node positions when plotting
            - sector_mask
                - function to mask an array
                - mask will be a sector of a circle
            - generate_node_positions
                - function to generate node positions
                    - used if no positions are provided to draw_MM
            - draw_MM
                - function to create an image of the MM
            - change_wallpaper
                - function to change a windows desktop wallpaper to a image provided by its absolute path
            - save
                - function to save a created MindMap to a text file
                - NOT IMPLEMENTED YET
            - load
                - function to load a MindMap saved as a text file
                - the file has to follow a specific structure (see documentation of load())

        Dependencies
        ------------
            - matplotlib
            - numpy
            - re
            - inspect
    """

    def __init__(
        self,
        node_contents=None, node_levels=None, hide_branch=None,  #node attributes
        edge_froms=None, edge_tos=None, edge_weights=None, edge_types=None, #edge attributes
        ):

        
        #nodes
        if node_contents is None:
            self.node_contents = np.array([], dtype=str)
        else:
            self.node_contents = np.array(node_contents)
        if node_levels is None:
            self.node_levels = np.zeros_like(self.node_contents, dtype=np.int16)
        else:
            self.node_levels = np.array(node_levels)
        if hide_branch is None:
            self.hide_branch = np.zeros_like(self.node_contents, dtype=bool)
        else:
            self.hide_branch = np.array(hide_branch, dtype=bool)

        #edges
        if edge_froms is None:
            self.edge_froms = np.array([], dtype=str)
        else:
            self.edge_froms = np.array(edge_froms)
        if edge_tos is None:
            self.edge_tos = np.empty_like(self.edge_froms, dtype=str)
        else:
            self.edge_tos = np.array(edge_tos)
        if edge_weights is None:
            self.edge_weights = np.ones_like(self.edge_froms, dtype=np.float16)
        else:
            self.edge_weights = np.array(edge_weights)
        if edge_types is None:
            self.edge_types = np.empty_like(self.edge_froms, dtype=str)
        else:
            self.edge_types = np.array(edge_types)

        self.nodes = {}
        self.edges = {}
        
        #check shapes and update self.nodes and self.edges
        self.get_nodes_edges()


    def get_nodes_edges(self, fix_shapes=False):
        """
            - function to get dicts of nodes and edges
        """
        import inspect
        import numpy as np
        
        
        #check shapes
        attributes = inspect.getfullargspec(self.__init__).args[1:]
        node_attributes = attributes[:int(attributes.index("edge_froms"))]
        edge_attributes = attributes[int(attributes.index("edge_froms")):]


        for na in node_attributes:
            #check if shapes are correct (exclude node_colors, becasue it is a tuple of 2 RGB values)
            if eval("self."+na+".shape") != self.node_contents.shape:
                if not fix_shapes:
                    raise ValueError(f"All node-attributes have to have the same shape. Shape of {na} {eval('self.'+na+'.shape')} does not match the other shapes {self.node_contents.shape}")

                #fix shapes by filling them with empties
                exec(f"self.{na} = np.empty_like(self.node_contents, dtype=self.{na}.dtype)")
            self.nodes[na] = eval("self."+na)

        for ea in edge_attributes:
            #check if shapes are correct
            if eval("self."+ea+".shape") != self.edge_froms.shape:
                if not fix_shapes:
                    raise ValueError(f"All edge-attributes have to have the same shape. Shape of {ea} {eval('self.'+ea+'.shape')} does not match the other shapes {self.edge_froms.shape}")
                
                #fix shapes by filling them with empties
                exec(f"self.{ea} = np.empty_like(self.edge_froms, dtype=self.{ea}.dtype)")
            self.edges[ea] = eval("self."+ea)


    def __repr__(self):

        return ("MindMap(\n"
                f"node_contents  = {repr(self.node_contents)},\n"
                f"node_levels    = {repr(self.node_levels)},\n"
                f"hide_branch    = {repr(self.hide_branch)},\n"
                f"edge_froms     = {repr(self.edge_froms)},\n"
                f"edge_tos       = {repr(self.edge_tos)},\n"
                f"edge_weights   = {repr(self.edge_weights)},\n"
                f"edge_typse     = {repr(self.edge_types)},\n"
                ")")


    #Error handling
    def check_keys(self, relationships={"from":[], "to":{}}, carac={"ID":[], "type":[]}):
        """
            - function to check if correct keys are provided to relationships and carac
        """
        #check if correct keys provided
        if ("from" or "to") not in relationships.keys():
            raise KeyError(f"""{relationships.keys()} are not valid keys for 'relationships'!
            'relationships' has to be of the following structure: {'{'}'from':[...], 'to':[...]{'}'} !""")
        if ("ID" or "type") not in carac.keys():
            raise KeyError(f"""{carac.keys()} are not valid keys for 'carac'!
            'carac' has to be of the following structure: {'{'}'ID':[...], 'type':[...]{'}'} !""")
        else:
            pass

    def generate_node_colors(self, cmap="jet", shift_min=0, shift_max=0):
        """
            - function to generate node colors according to the node level

            Parameters
            ----------
                - cmap
                    - str, optional
                    - matplotlib colormap name
                    - the default is 'jet'
                - shift_min
                    - int, optional
                    - value to shift vmin
                    - the default is 0  
                        - no shift
                - shift_max
                    - int, optional
                    - value to shift vmax
                    - the default is 0
                        - no shift
             Raises
             ------

             Returns
             -------
                - node_colors
                    - np.array
                    - contains rgb-tripels mapped to each node
            
            Dependencies
            ------------
                - matplotlib
                - numpy
        """

        #initialize with color-generator colors
        node_colors = np.empty((self.node_contents.shape[0], 3))
        if len(node_colors) > 0:
            if shift_max == 0: shift_max_to = None
            else: shift_max_to = -shift_max
            # level_colors = alp.Plot_LuSt.color_generator(ncolors=self.node_levels.max()+1, color2black_factor=.8, color2white_factor=1)[0]
            level_colors = eval(f"plt.cm.{cmap}(np.linspace(0,1,self.node_levels.max()+1+shift_min+shift_max))[shift_min:shift_max_to,:-1]")
            for lvl in np.unique(self.node_levels):
                node_colors[(lvl == self.node_levels)] = level_colors[lvl]
        
        return node_colors

    def get_fibonacci_disc(self, n, spacing=1):
        """
            - function to generate a fibonacci disc
            - http://blog.marmakoide.org/?p=1

            Parameters
            ----------
                - n
                    - int
                    - number of points to create
                - radial_offset
                    - float, optional
                    - offset in radial direction
                    - the default is 0
                - spacing
                    - float, optional
                    - amount of space between the points
                    - the default is 1

            Raises
            ------

            Returns
            -------
                - coords
                    - np.array
                    - 2d-array containing coordinates of the generated datapoints
            
            Dependencies
            ------------
                - numpy

            Comments
            --------

        """

        r = np.sqrt(np.arange(0, n, 1)/(n)).reshape((n, 1)) #radius of each point

        phi_golden = np.pi * (3 - np.sqrt(5)) #golden ratio
        phi = phi_golden * np.arange(0, n, 1) #polar angle

        #convert from polar coords to karthesian
        coords = np.zeros((n,2)) + r
        coords[:,0] *= np.cos(phi)
        coords[:,1] *= np.sin(phi)
        coords *= spacing

        return coords

    def sector_mask(self, coords, n_roots, center=[0,0], radius=1, start_angle=0, end_angle=None, testplot=False):
        """
            - function to divide 'coords' into 'n_roots' equally large partitions
                - each of the partitions is a sector of a circle with radius 'radius'
                - the circles origin is located at 'center'
            
            Parameters
            ----------
                - coords
                    - np.array
                    - 2d-array containing x- and y value tuples
                - n_roots
                    - int
                    - number of partitions to generate
                - center
                    - tuple, list, np.array, optional
                    - location of the origin of the circle, which is used to partition the data
                    - the default is [0,0]
                - radius
                    - float, optional
                    - radius of the circle, which is used to partition the data
                    - the default is 1
                - start_angle
                    - float, optional
                    - the angle to start partitioning at
                    - the default is 0
                - end_angle
                    - float, optional
                    - the angle to end partitioning at
                    - the default is None
                        - will lead to using a full circle (i.e. 2*pi)
                -testplot
                    - bool, optional
                    - whether to show a testplot
                    - the default is False                
            
            Raises
            ------

            Returns
            -------
                - coords_masks
                    - list
                        - contains 'n_roots' boolean masks
                - angle_range
                    - np.array
                    - range of angles used to partition the data
            
            Dependencies
            ------------
                - numpy
                - matplotlib
            
            Comments
            --------

        """
        
        if end_angle is None:
            end_angle = 2*np.pi


        #angles of partitions
        angle_range = np.linspace(start_angle, end_angle, n_roots+1)
        
        x = coords[:,0]
        y = coords[:,1]

        coords_masks = []
        for tmax, tmin in zip(-angle_range[:-1]+0.5*np.pi, -angle_range[1:]+0.5*np.pi):
            
            #carthesian to polar
            r2 = (x-center[0])**2 + (y-center[1])**2
            theta = np.arctan2(x-center[0], y-center[1]) - tmin
            theta %= 2*np.pi #wrap angles between 0 and 2*pi

            #radius mask
            rad_mask = (r2 <= radius**2)
            #angular mask
            angle_mask = (theta <= (tmax-tmin))

            coords_masks.append(rad_mask*angle_mask)
        
        if testplot:
            plt.figure(figsize=(10,10))
            for cm in coords_masks:
                plt.scatter(coords[cm][:,0], coords[cm][:,1])
            plt.show()


        return coords_masks, angle_range

    def generate_node_positions(self, 
        section_separation=0.1, spacing=1,
        center=[0,0], start_angle=0, end_angle=None,
        correction_exponent=2,
        ):
        #TODO: not working as expected for depth >=4
        """
            - function to generate some default node-positions
            - will generate positions consisting of datapoints on a fibonacci disc with radius 1 around 'center'
            - will generate sectors corresponding to the number of root nodes (nodes with 'node_level' == 0)

            Parameters
            ----------
                - section_separation
                    - float, optional
                    - measure of how much each sector shall be separated from the other ones
                    - the default is 0.1
                - spacing
                    - float, optional
                    - amount of space between the points
                    - the default is 1
                - center
                    - tuple, list, np.array, optional
                    - location of the origin of the circle, which is used to partition the data
                    - the default is [0,0]
                - start_angle
                    - float, optional
                    - the angle to start partitioning the fibonacci disc at
                    - the default is 0
                - end_angle
                    - float, optional
                    - the angle to end partitioning the fibonacci disc at
                    - the default is None
                        - will lead to using a full circle (i.e. 2*pi)
                - correction_exponenet
                    - float, optional
                    - exponent to generate more datapoints
                    - needed in case one branch has a lot of nodes
                        - thus, even partitioning will lead to an IndexError
                    - the default is 2

            Raises
            ------

            Returns
            -------
                - node_positions
                    - np.array
                    - 2d
                    - coordinates for each node in the MindMap
            
            Dependencies
            ------------
                - numpy
            
            Comments
            --------
        """

        n_roots = np.count_nonzero((self.node_levels == 0))
        n_nodes = self.node_contents.shape[0]

        #generate coordinates
        coords = self.get_fibonacci_disc(int(n_nodes**correction_exponent), spacing=spacing)
        divisions, angle_range = self.sector_mask(
            coords, n_roots, center=center,
            radius=spacing, start_angle=start_angle, end_angle=end_angle,
            testplot=False)

        #generate shifts to separate sections
        shifts = section_separation*np.array([
            np.cos(angle_range),
            np.sin(angle_range)
            ]).T

        #initiate node positions
        node_positions = np.zeros(shape=(self.node_contents.shape[0],2))
        
        min_level = self.node_levels.min()
        if min_level != 0:
            print("Make sure that at least one of your node_leves == 0. If this is not the case, the plotting might lead to unwanted results.")

        #set root nodes (used as starting point for main branches)
        for d, s, root, in zip(divisions, shifts, self.node_contents[(self.node_levels==min_level)]):

            coords[d] = coords[d]+s #update coordinates for each sector with the sector separation
            

            distance_sort = np.argsort((coords[d][:,0]**2 + coords[d][:,1]**2)) #sort each sector by distance to (0,0)

            root_coords = coords[d][distance_sort[0]]   #coordinates for the root note - point closest to (0,0) of the respective sector
            node_positions[(self.node_contents == root)] = root_coords      #set node_position for root to root_coords


        #set coordinates for all other nodes
        #iterate until deepest element is reached
        max_depth = self.node_levels.max()
        d = min_level
        while d <= max_depth:
            cur_ids = self.node_contents[(self.node_levels==d)]

            #iterate over all elements in the current depth
            for idx, cur_id in enumerate(cur_ids):
                from_to_idx = (self.edge_froms == cur_id)

                #iterate over all edges from current depth to next depth
                for idx_sub, (ef, et) in enumerate(zip(self.edge_froms[from_to_idx], self.edge_tos[from_to_idx])):
                    mask_sub, angle_range = self.sector_mask(
                        coords, 1, center=node_positions[(self.node_contents == ef)][0],
                        radius=spacing, start_angle=0, end_angle=2*np.pi,
                        testplot=False
                    )
                    distance_sort_sub = np.argsort((coords[mask_sub[0]][:,0] - node_positions[(self.node_contents == ef)][0,0])**2 + (coords[mask_sub[0]][:,1] - node_positions[(self.node_contents == ef)][0,1])**2 )
                    

                    #check if shortest distance is already taken
                    #if so, check the next shortest
                    #repeat until one distance is not taken
                    checkidx = 0
                    while coords[mask_sub[0]][distance_sort_sub[checkidx]] in node_positions:
                        checkidx += 1
                    node_positions[(self.node_contents == et)] = coords[mask_sub[0]][distance_sort_sub[checkidx]]

            d+=1    #proceed to next depth
        
        return node_positions


    #MM operations
    def draw_MM(self,
        node_positions=None, node_sizes=1, node_colors="jet", node_shape="o", node_borderwidth=0, node_bordercolor="k",     #Nodes
        shift_vmin=0, shift_vmax=0,
        edge_colors="k", edge_styles="-",
        font_size=12, font_color="k",
        plt_style="default",
        section_separation=0.1, spacing=1,
        start_angle=0, end_angle=None,
        correction_exponent=2,
        xmargin=0.15, ymargin=0.15, pad=1.08,
        figsize=(16,9), ax_pos=111,
        save=False, dpi=180):               
        """
            - function to show the created Mind-Map

            Parameters
            ----------
                - node_sizes
                    - list, int, optional
                    - list of sizes for each node
                    - the default is None
                        - will outogenereate a size according to the specified "level" of a node
                        - L0 > L1 > L2 > ...
                - node_colors
                    - list, str, optional
                    - list of colors for the nodes
                    - str of a matplotlib colormap-name
                    - the default is 'jet'
                        - Colors generated according to a matplotlib colormap
                        - will generate colors, according to the "level" of each node
                        - L1: Color1, L2: Color2, ...
                - node_shape
                    - str, optional
                    - defines shape of the nodes
                    - uses matplotlib markers
                    - the default is "o"
                - node_borderwidth
                    - float, optional
                    - defines the width of the border surrounding a node
                    - the default is 0, i.e. no border
                - node_bordercolor
                    - str, rgb-triple optional
                    - defines color of the border surrounding a node
                    - the default is black
                - shift_vmin
                    - int, optional
                    - value to shift vmin
                    - the default is 0  
                        - no shift
                - shift_vmax
                    - int, optional
                    - value to shift vmax
                    - the default is 0
                        - no shift                
                - edge_colors
                    - list, str, optional
                    - list of colors for all the edges
                    - the default is "black"
                - edge_styles
                    - list, str, optional
                    - list of the styles to use for each edge
                    - uses matplotlib linestyle strings
                    - the default is a solid line
                - font_size
                    - int, optional
                    - font size of the IDs (content) of each node
                    - the default is 12
                - font_color
                    - str, optional
                    - color of the IDs (content) of each node
                    - the default is "black"
                - plt_style
                    - str, optional
                    - which style of matplotlib to use for plotting
                    - the default is "default"
                - section_separation
                    - float, optional
                    - measure of how much each branch shall be separated from the other ones
                    - the default is 0.1
                - spacing
                    - float, optional
                    - amount of space between the nodes
                    - the default is 1
                - center
                    - tuple, list, np.array, optional
                    - location of the origin of the circle, which is used to partition the datapoints
                    - the default is [0,0]
                - start_angle
                    - float, optional
                    - the angle to start partitioning the fibonacci disc at
                    - the default is 0
                - end_angle
                    - float, optional
                    - the angle to end partitioning the fibonacci disc at
                    - the default is None
                        - will lead to using a full circle (i.e. 2*pi)
                - correction_exponenet
                    - float, optional
                    - exponent to generate more datapoints
                    - needed in case one branch has a lot of nodes
                        - thus, even partitioning will lead to an IndexError
                    - the default is 2
                - xmargin
                    - float, optional
                    - some value to specify the distance of the data towards the borders
                    - the default is 0.15
                - ymargin
                    - float, optional
                    - some value to specify the distance of the data towards the borders
                    - the default is 0.15
                - pad
                    - int, optional
                    - padding in plt.tight_layout()
                    - high values will push the datapoints closer together
                    - low values will pull the datapoints further apart
                    - the default 1.08
                - figsize
                    - tuple, optional
                    - size of the figure
                    - the default is (16,9)
                - ax_pos
                    - int, optional
                    - matplotlib axis position specifier
                    - position of the axis used for the MindMap
                    - useful if you want to include something else in the figure as well
                    - the default is '111'
                        - will fill the whole figure
                - save
                    - str, bool, optional
                    - whether to save the produced output
                        - will be saved if a string is provided
                        - will be saved to that respective location
                    - the default is False
                - dpi
                    - int, optional
                    - the resolution to use for saving the image
                    - the default is 180

            Raises
            ------
                - IndexError
                    - in case too little nodes for a particular branch have been generated
                    - also gives the hint of increasing 'correction_exponen'

            Returns
            -------
                - fig
                    - matplotlib figure object
                - ax
                    - matplotlib axes object

            Dependencies
            ------------
                - matplotlib

        """

        #change background        
        plt.style.use(plt_style)

        #generate node_positions
        if node_positions is None:
            try:
                node_positions = self.generate_node_positions(
                    section_separation=section_separation, spacing=spacing,
                    center=[0,0], start_angle=start_angle, end_angle=end_angle,
                    correction_exponent=correction_exponent,
                )
            except IndexError as i:
                print(f"ORIGINAL ERROR: {i}")
                raise IndexError("Try to increase 'correction_exponent' in order to ensure not falling out of bounds.")
        if type(node_colors) == str:
            #treat node_colors differently, since it can be an array of RGB tripels
            node_colors = self.generate_node_colors(cmap=node_colors, shift_min=shift_vmin, shift_max=shift_vmax)


        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(ax_pos)
        ax.scatter(
            node_positions[:,0], node_positions[:,1],
            s=node_sizes*5000/(self.node_levels+1), c=node_colors, marker=node_shape,
            linewidth=node_borderwidth, edgecolor=node_bordercolor,
            zorder=1
            )
        for nps, content in zip(node_positions, self.node_contents):
            ax.text(
                nps[0], nps[1], content,
                horizontalalignment="center", verticalalignment="center", multialignment="center",
                fontsize=font_size, color=font_color,
                )
        for ef, et in zip(self.edge_froms, self.edge_tos):
            np_from = node_positions[(ef == self.node_contents)]
            np_to   = node_positions[(et == self.node_contents)]
            ax.plot((np_from[0,0], np_to[0,0]), (np_from[0,1], np_to[0,1]), linestyle=edge_styles, color=edge_colors, zorder=0)
        
        #set axis margins (to push datapints around in the frame)
        ax.margins(xmargin, ymargin)

        #hide axes
        plt.axis("off")
        plt.tight_layout(pad=pad)

        #save
        if type(save) == str:
            plt.savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0)

        return fig, ax

    def change_wallpaper(self, absolute_path):
        """
            - function to change the windows-wallpaper to your created mindmap

            Parameters
            ----------
                - absolute_path
                    - str
                    - the ABSOLUTE path to the source file

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - ctypes
            
            Comments
            --------

        """
            
        #change wallpaper
        try:
            ctypes.windll.user32.SystemParametersInfoW(20, 0, absolute_path , 0)
        except Exception as e:
            print(f"ORIGINAL ERROR: {e}")
            raise ValueError("Probaly you have to change the path to your list of TODOs")
        
        pass


    #saving and loading
    def save(self, path="MM_temp.txt"):
        #TODO: Implement - use code from generate_node_positions?
        #TODO: Make sure to leave one empty line at the start of the document!
        """
            - function to save the MindMap as an indented list in a text-file
        """

        max_depth = self.node_levels.max()
        min_depth = self.node_levels.min()
        print()


        roots = self.node_contents[(self.node_levels==min_depth)]
        for root in roots:
            print(root)
            connection_idxs = (self.edge_froms==root)

            while np.any(connection_idxs):
                for c_sub in self.edge_tos[connection_idxs]:
                    print("\t"*self.node_levels[(self.node_contents==c_sub)][0], c_sub)
                    connection_idxs_sub = (self.edge_froms==c_sub)

                    for c_2sub in self.edge_tos[connection_idxs_sub]:
                        print("\t"*self.node_levels[(self.node_contents==c_2sub)][0], c_2sub)

                    connection_idxs = (self.edge_froms==c_sub)


        raise NotImplementedError("NOT IMPLEMENTED YET!")
        pass

    def load(self, path):
        """
            - function to load MindMap from an indented bullet-list in a md-file
            - each node has to be lead by '- '
            - an example would be: \n
                >>> - Parent1  \n
                >>>     - Child11 \n
                >>>         - Child111 \n
                >>>     - Child12 \n
                >>>         - Child121 \n
                >>>         - Child122 \n
                >>> - Parent2 \n
                >>>     - Child21 \n
                >>>     - ... \n
            
            Parameters
            ----------
                - path
                    - str
                        - the path to the source file
                        - has to be of above mentioned structure

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - numpy
                - re
            
            Comments
            --------
        """
        import numpy as np

        # raise Warning("Not implement yet!")
        infile = open(path)
        content = infile.read()
        infile.close()

        entries = re.findall(r"(?<=- \b).+", content)
        indents = re.findall(r"^[^\S\n]*(?=-)", content, flags=re.M)

        #add some element which will be ignored
        #only needed to ensure that the last actual element gets used in the MindMap
        entries_ = np.append(entries, "IGNORE")
        indents_ = np.append(indents, "")

        #get self.carac ('ID's and 'type's)
        self.node_contents = np.array([e for e in entries_])
        self.node_levels = np.array([len(i)//4 for i in indents_[:len(entries_)]])

        max_depth = self.node_levels.max()                      #deepest element
        d = 0                                                   #initiate at root node to get into loop
        #iterate until the deepest element is reached
        while d <= max_depth:
            cur_ids = self.node_contents[(self.node_levels==d)]
            sub_ids = self.node_contents[(self.node_levels==(d+1))]

            #iterate over all elements in the current depth (get parents)
            for cur_id, next_id in zip(cur_ids[:-1], cur_ids[1:]):
                next_idx = int(np.where(self.node_contents == next_id)[0])
                cur_idx = int(np.where(self.node_contents == cur_id)[0])
                
                #iterate over all elements in the next depth (get children)
                for sub_id in sub_ids:
                    sub_idx = int(np.where(self.node_contents == sub_id)[0])

                    #check if the children are before the next and after the previous parent
                    #i.e.: check if the children belong to the respective layer
                    if cur_idx < sub_idx and sub_idx < next_idx:
                        
                        #append to the respective data-structures
                        self.edge_froms = np.append(self.edge_froms, cur_id)
                        self.edge_tos = np.append(self.edge_tos, sub_id)
            
            #proceed in next depth
            d += 1
        
        #discard "IGNORE"-element
        self.node_contents = self.node_contents[:-1]
        self.node_levels = self.node_levels[:-1]

        #update attributes
        self.get_nodes_edges(fix_shapes=True)

        pass