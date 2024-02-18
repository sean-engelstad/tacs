#!/usr/bin/python
"""
pymeshloader - A python-based mesh loader for setting up TACS

This python interface is designed to provide an easier interface to the
c-layer of TACS. This module uses the pyNastran library for its bdf parsing
functionality.

"""
# =============================================================================
# Imports
# =============================================================================
from copy import deepcopy
import itertools as it

import numpy as np
import pyNastran.bdf.bdf as pn

import tacs.TACS
import tacs.constitutive
import tacs.elements
from tacs.utilities import BaseUI


class pyMeshLoader(BaseUI):
    def __init__(self, comm, printDebug=False):
        # Set MPI communicator
        BaseUI.__init__(self, comm=comm)
        # Debug printing flag
        self.printDebug = printDebug
        self.bdfInfo = None

    def scanBdfFile(self, bdf):
        """
        Scan nastran bdf file using pyNastran's bdf parser.
        We also set up arrays that will be required later to build tacs.
        """

        # Only print debug info on root, if requested
        if self.comm.rank == 0:
            debugPrint = self.printDebug
        else:
            debugPrint = False

        # Check if a file name was provided
        # Create a BDF object from the file
        if isinstance(bdf, str):
            # Read in bdf file as pynastran object
            # By default we avoid cross-referencing unless we actually need it,
            # since its expensive for large models
            self.bdfInfo = pn.read_bdf(
                bdf, validate=False, xref=False, debug=debugPrint
            )
        # Create a copy of the BDF object
        elif isinstance(bdf, pn.BDF):
            self.bdfInfo = deepcopy(bdf)
        else:
            raise self._TACSError(
                "BDF input must be provided as a file name 'str' or pyNastran 'BDF' object. "
                f"Provided input was of type '{type(bdf).__name__}'."
            )

        # Set flag letting us know model is not xrefed yet
        self.bdfInfo.is_xrefed = False

        # If any property cards are missing in the bdf, we have to add dummy cards
        # so pynastran doesn't through errors when cross-referencing
        # Loop through all elements and add dummy property, as necessary
        self.bdfInfo.missing_properties = False
        for element_id in self.bdfInfo.elements:
            element = self.bdfInfo.elements[element_id]
            if element.pid not in self.bdfInfo.property_ids:
                # If no material properties were found,
                # add dummy properties and materials
                matID = 1
                E = 70.0
                G = 35.0
                nu = 0.3
                self.bdfInfo.add_mat1(matID, E, G, nu)
                self.bdfInfo.add_pbar(element.pid, matID)
                # Warn the user that the property card is missing
                # and should not be read in using pytacs elemCallBackFromBDF method
                self.bdfInfo.missing_properties = True
                if self.printDebug:
                    self._TACSWarning(
                        "Element ID %d references undefined property ID %d in bdf file. "
                        "A user-defined elemCallBack function will need to be provided."
                        % (element_id, element.pid)
                    )

        # We have to remove any empty property groups that may have been read in from the BDF
        self.propertyIDToElementIDDict = (
            self.bdfInfo.get_property_id_to_element_ids_map()
        )
        for pid in self.propertyIDToElementIDDict:
            # If there are no elements referencing this property card, remove it
            if len(self.propertyIDToElementIDDict[pid]) == 0:
                self.bdfInfo.properties.pop(pid)
                self.propertyIDToElementIDDict.pop(pid)

        # Create dictionaries for mapping between tacs and nastran id numbering
        self._updateNastranToTACSDicts()

        # Try to get the node x,y,z locations from bdf file
        try:
            self.bdfXpts = self.bdfInfo.get_xyz_in_coord(
                fdtype=self.dtype, sort_ids=False
            )
        # If this fails, the file may reference multiple coordinate systems
        # and will have to be cross-referenced to work
        except:
            self.bdfInfo.cross_reference()
            self.bdfInfo.is_xrefed = True
            self.bdfXpts = self.bdfInfo.get_xyz_in_coord(
                fdtype=self.dtype, sort_ids=False
            )

        # element card contained within each property group (may contain multiple per group)
        # Each entry will eventually have its own tacs element object assigned to it
        # (ex. self.elemDescripts = [['CQUAD4', 'CTRIA3'], ['CQUAD9', 'CQUAD4'], ['CBAR']])
        self.elemDescripts = []
        # Pointer index used to map elemDescript entries into flat list
        # (ex. self.elementObjectNumByComp = [[0, 1], [2, 3], [4]])
        self.elemObjectNumByComp = []
        # User-defined name for property/component group,
        # this may be read in through ICEMCFD/FEMAP format comments in BDF
        self.compDescripts = []

        # Populate list entries with default values
        for pID in self.bdfInfo.property_ids:
            self.elemDescripts.append([])
            self.elemObjectNumByComp.append([])
            # Check if there is a Femap/HyperMesh/Patran label for this component
            propComment = self.bdfInfo.properties[pID].comment
            # Femap format
            if "$ Femap Property" in propComment:
                # Pick off last word from comment, this is the name
                propName = propComment.split()[-1]
                self.compDescripts.append(propName)
            # HyperMesh format
            elif "$HMNAME PROP" in propComment:
                # Locate property name line
                loc = propComment.find("HMNAME PROP")
                compLine = propComment[loc:]
                # The component name is between double quotes
                propName = compLine.split('"')[1]
                self.compDescripts.append(propName)
            # Patran format
            elif "$ Elements and Element Properties for region" in propComment:
                # The component name is after the colon
                propName = propComment.split(":")[1]
                self.compDescripts.append(propName)

            # No format, default component name
            else:
                self.compDescripts.append(f"Property group {pID}")

        # Element connectivity information
        self.elemConnectivity = [None] * self.bdfInfo.nelements
        self.elemConnectivityPointer = [None] * (self.bdfInfo.nelements + 1)
        self.elemConnectivityPointer[0] = 0
        elementObjectCounter = 0
        # List specifying which tacs element object each element in bdf should point to
        self.elemObjectNumByElem = [None] * (self.bdfInfo.nelements)

        # Loop through every element and record information needed for tacs
        for tacsElementID, nastranElementID in enumerate(self.bdfInfo.element_ids):
            element = self.bdfInfo.elements[nastranElementID]
            elementType = element.type.upper()
            propertyID = element.pid
            componentID = self.idMap(propertyID, self.nastranToTACSCompIDDict)

            # This element type has not been added to the list for the component group yet, so we append it
            if elementType not in self.elemDescripts[componentID]:
                self.elemDescripts[componentID].append(elementType)
                self.elemObjectNumByComp[componentID].append(elementObjectCounter)
                elementObjectCounter += 1

            # Find the index number corresponding to the element object number for this component
            componentTypeIndex = self.elemDescripts[componentID].index(elementType)
            self.elemObjectNumByElem[tacsElementID] = self.elemObjectNumByComp[
                componentID
            ][componentTypeIndex]

            # We've identified a ICEM property label
            if "Shell element data for family" in element.comment:
                componentName = element.comment.split()[-1]
                self.compDescripts[componentID] = componentName

            conn = element.nodes.copy()

            # TACS has a different node ordering than Nastran for certain elements,
            # we now perform the reordering (if necessary)
            if elementType in ["CQUAD4", "CQUADR"]:
                conn = [conn[0], conn[1], conn[3], conn[2]]
            elif elementType in ["CQUAD9", "CQUAD"]:
                conn = [
                    conn[0],
                    conn[4],
                    conn[1],
                    conn[7],
                    conn[8],
                    conn[5],
                    conn[3],
                    conn[6],
                    conn[2],
                ]
            elif elementType in ["CHEXA8", "CHEXA"]:
                conn = [
                    conn[0],
                    conn[1],
                    conn[3],
                    conn[2],
                    conn[4],
                    conn[5],
                    conn[7],
                    conn[6],
                ]

            # Map node ids in connectivity from Nastran numbering to TACS numbering
            self.elemConnectivity[tacsElementID] = self.idMap(
                conn, self.nastranToTACSNodeIDDict
            )
            self.elemConnectivityPointer[
                tacsElementID + 1
            ] = self.elemConnectivityPointer[tacsElementID] + len(element.nodes)

        # Allocate list for user-specified tacs element objects
        self.elemObjects = [None] * elementObjectCounter

        # Total number of nodes used to hold lagrange multiplier variables
        self.numMultiplierNodes = 0
        # List to hold ID numbers (TACS ordering) of multiplier nodes added to the problem later
        self.multiplierNodeIDs = []

    def _updateNastranToTACSDicts(self):
        """
        Create dictionaries responsible for mapping over
        global node, global element, and property/component ID numbers
        from NASTRAN (as read in from BDF) to TACS (contiguous, 0-based).
        The keys of each dictionary are the NASTRAN ID and the entry the TACS ID,
        such that:
            tacsNodeID = self.nastranToTACSNodeIDDict[nastranNodeID]
            tacsComponentID = self.nastranToTACSCompIDDict[nastranPropertyID]
            tacsElementID = self.nastranToTACSNodeIDDict[nastranElementID]
        The dictionaries contain all elements/nodes found in the BDF,
        not just those *owned* by this processor
        """
        # Create Node ID map
        nastranIDs = self.bdfInfo.node_ids
        tacsIDs = range(self.bdfInfo.nnodes)
        nodeTuple = zip(nastranIDs, tacsIDs)
        self.nastranToTACSNodeIDDict = dict(nodeTuple)

        # Create Property/Component ID map
        nastranIDs = self.bdfInfo.property_ids
        tacsIDs = range(self.bdfInfo.nproperties)
        propTuple = zip(nastranIDs, tacsIDs)
        self.nastranToTACSCompIDDict = dict(propTuple)

        # Create Element ID map
        nastranIDs = self.bdfInfo.element_ids
        tacsIDs = range(self.bdfInfo.nelements)
        elemTuple = zip(nastranIDs, tacsIDs)
        self.nastranToTACSElemIDDict = dict(elemTuple)

    def getBDFInfo(self):
        """
        Return pynastran bdf object.

        Returns
        -------
        bdfInfo : pyNastran.bdf.bdf.BDF
            pyNastran bdf object.
        """
        return self.bdfInfo

    def getNumComponents(self):
        """
        Return number of components (properties) found in bdf.

        Returns
        -------
        nComps : int
            Number of components (properties) found in bdf file.
        """
        return self.bdfInfo.nproperties

    def getNumBDFNodes(self):
        """
        Return number of nodes found in bdf.

        Returns
        -------
        nNodes : int
            Number of nodes found in bdf file.
        """
        return self.bdfInfo.nnodes

    def getNumOwnedNodes(self):
        """
        Return number of nodes owned by this processor.

        Returns
        -------
        nNodes : int
            Number of nodes owned by this proc.
        """
        return self.assembler.getNumOwnedNodes()

    def getNumBDFElements(self):
        """
        Return number of elements found in bdf.

        Returns
        -------
        nElems : int
            Number of elements found in bdf file.
        """
        return self.bdfInfo.nelements

    def getBDFNodes(self, nodeIDs, nastranOrdering=False):
        """
        Return x,y,z location of specified node in bdf file.

        Returns
        -------
        xyz : numpy.ndarray
            Coordinates of specified nodes.
        """
        # Convert to tacs numbering, if necessary
        if nastranOrdering:
            nodeIDs = self.idMap(nodeIDs, self.nastranToTACSNodeIDDict)
        return self.bdfXpts[nodeIDs]

    def getElementComponents(self):
        """
        Get a list specifying the component ID of each element.

        Returns
        -------
        compIDList : list[int]
            List containing componentID of each element found in the bdf file.
        """
        elements = self.bdfInfo.elements
        propertyIDList = [elements[eID].pid for eID in self.bdfInfo.element_ids]
        compIDList = self.idMap(propertyIDList, self.nastranToTACSCompIDDict)
        return compIDList

    def getConnectivityForComp(self, componentID, nastranOrdering=False):
        """
        Get a nodal connectivities of each element belonging to the specified component.

        Parameters
        ----------
        componentID : int
            Component ID number.

        nastranOrdering : bool
            Flag signaling whether nodeIDs should be returned in TACS (default) or NASTRAN (grid IDs in bdf file) ordering
            Defaults to False.

        Returns
        -------
        elemConns : list[list[int]]
            List of nodal connectivities of each element belonging to this component.
        """
        # Find all of the element IDs belonging to this property group
        propertyID = list(self.bdfInfo.property_ids)[componentID]
        elementIDs = self.propertyIDToElementIDDict[propertyID]
        compConn = []
        for elementID in elementIDs:
            # We've now got the connectivity for this element, but it is in nastrans node numbering
            nastranConn = self.bdfInfo.elements[elementID].nodes
            if nastranOrdering:
                compConn.append(nastranConn)
            else:
                # Convert Nastran node numbering back to tacs numbering
                tacsConn = self.idMap(nastranConn, self.nastranToTACSNodeIDDict)
                # Append element connectivity to list for component
                compConn.append(tacsConn)
        return compConn

    def getElementObjectNumsForComp(self, componentID):
        """
        Return tacs element object number for each element type
        belonging to this component.

        Parameters
        ----------
        componentID : int
            Component ID number.

        Returns
        -------
        objNums : list[int]
            List holding element object nums for the specifed component.
        """
        return self.elemObjectNumByComp[componentID][:]

    def getElementDescripts(self):
        """
        Get nested list containing all element types owned by each component group
        example: [['CQUAD4', 'CTRIA3], ['CQUAD4'], ['CQUAD4', CQUAD9']]

        Returns
        -------
        elemDescripts : list[list[str]]
            Nested list holding all element types owned by each component group.
        """
        return self.elemDescripts

    def getComponentDescripts(self):
        """
        Get user-defined labels for each component read in from the BDF.

        Returns
        -------
        compDescripts : list[str]
            List holding description strings for each component.
        """
        return self.compDescripts

    def getLocalNodeIDsFromGlobal(self, globalIDs, nastranOrdering=False):
        """
        Given a list of node IDs in global (non-partitioned) ordering
        returns the local (partitioned) node IDs on each processor.
        If a requested node is not included on this processor,
        an entry of -1 will be returned.

        Parameters
        ----------
        globalIDs : int or list[int]
            List of global node IDs.

        nastranOrdering : bool
            Flag signaling whether globalIDs is in TACS (default) or NASTRAN (grid IDs in bdf file) ordering
            Defaults to False.

        Returns
        -------
        localIDs : list[int]
            List of local node IDs for each entry in globalIDs.
            If the node is not owned by this processor, its index is filled with a value of -1.
        """
        # Convert to tacs node numbering, if necessary
        if nastranOrdering:
            globalIDs = self.idMap(globalIDs, self.nastranToTACSNodeIDDict)

        # Ensure input is list-like
        globalIDs = np.atleast_1d(globalIDs)

        # Get the node id offset for this processor
        OwnerRange = self.assembler.getOwnerRange()
        nodeOffset = OwnerRange[self.comm.rank]

        # Get the local ID numbers for this proc
        tacsLocalIDs = []
        for gID in globalIDs:
            lIDs = self.creator.getAssemblerNodeNums(
                self.assembler, np.array([gID], dtype=np.intc)
            )
            # Node was not found on this proc, return -1
            if len(lIDs) == 0:
                tacsLocalIDs.append(-1)
            # Node was found on this proc, shift by nodeOffset to get local index for node
            else:
                tacsLocalIDs.append(lIDs[0] - nodeOffset)

        return tacsLocalIDs

    def getLocalElementIDsFromGlobal(self, globalIDs, nastranOrdering=False):
        """
        Given a list of element IDs in global (non-partitioned) ordering
        returns the local (partitioned) element IDs on each processor.
        If a requested element is not included on this processor,
        an entry of -1 will be returned.

        Parameters
        ----------
        globalIDs : int or list[int]
            List of global element IDs.

        nastranOrdering : bool
            Flag signaling whether globalIDs is in TACS (default) or NASTRAN (element IDs in bdf file) ordering
            Defaults to False.

        Returns
        -------
        localIDs : list[int]
            List of local element IDs for each entry in globalIDs.
            If the element is not owned by this processor, its index is filled with a value of -1.
        """
        # Convert to tacs node numbering, if necessary
        if nastranOrdering:
            globalIDs = self.idMap(globalIDs, self.nastranToTACSElemIDDict)

        # Ensure input is list-like
        globalIDs = np.atleast_1d(globalIDs)

        # Get the local ID numbers for this proc
        tacsLocalIDs = []
        for gID in globalIDs:
            # element was found on this proc, get local ID num
            if gID in self.globalToLocalElementIDDict:
                lID = self.globalToLocalElementIDDict[gID]
            # element was not found on this proc, return -1
            else:
                lID = -1
            tacsLocalIDs.append(lID)

        return tacsLocalIDs

    def getGlobalNodeIDsForComps(self, componentIDs, nastranOrdering=False):
        """
        Return the global (non-partitioned) node IDs belonging to a given list of component IDs

        Parameters
        ----------
        componentIDs : int or list[int]
            List of integers of the compIDs numbers.

        nastranOrdering : bool
            Flag signaling whether nodeIDs should be returned in TACS (default) or NASTRAN (grid IDs in bdf file) ordering
            Defaults to False.

        Returns
        -------
        nodeIDs : list[int]
            List of unique nodeIDs that belong to the given list of compIDs
        """
        # First determine the actual physical locations
        # of the nodes we want to add forces
        # to. Only the root rank need do this:
        uniqueNodes = None
        if self.comm.rank == 0:
            allNodes = []
            componentIDs = self._flatten(componentIDs)
            componentIDs = set(componentIDs)
            for cID in componentIDs:
                tmp = self.getConnectivityForComp(cID, nastranOrdering=nastranOrdering)
                allNodes.extend(self._flatten(tmp))

            # Now just unique all the nodes:
            uniqueNodes = np.unique(allNodes)

        uniqueNodes = self.comm.bcast(uniqueNodes, root=0)

        return list(uniqueNodes)

    def getLocalNodeIDsForComps(self, componentIDs):
        """
        return the local (partitioned) node IDs belonging to a given list of component IDs

        Parameters
        ----------
        componentIDs : int or list[int]
            List of integers of the compIDs numbers.

        Returns
        -------
        nodeIDs : list[int]
            List of unique nodeIDs that belong to the given list of compIDs
        """
        # Get the global nodes for this component (TACS ordering)
        globalNodeIDs = self.getGlobalNodeIDsForComps(
            componentIDs, nastranOrdering=False
        )

        # get the corresponding local IDs on each proc
        localNodeIDs = self.getLocalNodeIDsFromGlobal(
            globalNodeIDs, nastranOrdering=False
        )
        # If the returned index is > 0 then it is owned by this proc, otherwise remove it
        localNodeIDs = [localID for localID in localNodeIDs if localID >= 0]

        return localNodeIDs

    def getGlobalElementIDsForComps(self, componentIDs, nastranOrdering=False):
        """
        Returns a list of element IDs belonging to specified components

        Parameters
        ----------
        componentIDs : list[int]
            Component ID numbers.

        nastranOrdering : bool
            Flag signaling whether elemIDs should be returned in TACS (default) or NASTRAN (grid IDs in bdf file) ordering
            Defaults to False

        Returns
        -------
        elemIDs : list[int]
            List of global element IDs belonging to the specified components.
        """
        # Make sure list is flat
        componentIDs = self._flatten(componentIDs)
        # Convert tacs component IDs to nastran property IDs
        propertyIDs = [0] * len(componentIDs)
        for i, componentID in enumerate(componentIDs):
            propertyIDs[i] = list(self.bdfInfo.property_ids)[componentID]
        # Get dictionary whose values are the element ids we are looking for
        elementIDDict = self.bdfInfo.get_element_ids_dict_with_pids(propertyIDs)
        # Convert to list
        elementIDs = list(elementIDDict.values())
        # Make sure list is flat
        elementIDs = self._flatten(elementIDs)
        # Convert to tacs element numbering, if necessary
        if not nastranOrdering:
            elementIDs = self.idMap(elementIDs, self.nastranToTACSElemIDDict)
        return elementIDs

    def getLocalElementIDsForComps(self, componentIDs):
        """
        Get the local element numbers on each proc used by tacs
        corresponding to the component groups in componentIDs.

        Parameters
        ----------
        componentIDs : list[int]
            Component ID numbers.

        Returns
        -------
        elemIDs : list[int]
            List of local element IDs on this proc belonging to the specified components.
        """
        if self.creator is None:
            raise self._TACSError(
                "TACS assembler has not been created. "
                "Assembler must created first by running 'createTACS' method."
            )
        # Make sure list is flat
        componentIDs = self._flatten(componentIDs)
        # Get the element object IDs belonging to each of these components
        objIDs = []
        for componentID in componentIDs:
            tmp = self.getElementObjectNumsForComp(componentID)
            objIDs.extend(tmp)
        objIDs = np.array(objIDs, dtype=np.intc)
        # Get the local element IDs corresponding to the object IDs on this processor (if any)
        localElemIDs = self.creator.getElementIdNums(objIDs)
        return list(localElemIDs)

    def getLocalMultiplierNodeIDs(self):
        """
        Get the tacs indices of multiplier nodes used to hold lagrange multipliers on this processor.

        Returns
        -------
        nodeIDs : list[int]
            List of multiplier node ID's owned by this proc.
        """
        return self.ownedMultiplierNodeIDs

    def getGlobalToLocalNodeIDDict(self):
        """
        Creates a dictionary who's keys correspond to the global ID of each node (tacs ordering)
        owned by this processor and whose values correspond to the local node index for this proc.
        The dictionary can be used to convert from a global node ID to local using the assignment below*:
        localNodeID = globalToLocalNodeIDDict[globalNodeID]

        * assuming globalNodeID is owned on this processor

        Returns
        -------
        globalToLocalNodeIDDict : dict[int,int]
            Dictionary holding mapping from global to local node IDs for this proc
        """
        globalToLocalNodeIDDict = {}
        for tacsNodeID in range(self.bdfInfo.nnodes):
            # Get the local ID corresponding to the global ID (if owned by this proc)
            lID = self.getLocalNodeIDsFromGlobal(tacsNodeID, nastranOrdering=False)[0]
            # Add the local node ID to the dict if its owned by this proc
            if lID >= 0:
                globalToLocalNodeIDDict[tacsNodeID] = lID

        return globalToLocalNodeIDDict

    def getGlobalToLocalElementIDDict(self):
        """
        Creates a dictionary who's keys correspond to the global ID of each element (tacs ordering)
        owned by this processor and whose values correspond to the local element index for this proc.
        The dictionary can be used to convert from a global element ID to local using the assignment below*:
        localElementID = globalToLocalElementIDDict[globalElementID]

        * assuming globalElementID is owned on this processor

        Returns
        -------
        globalToLocalElementIDDict : dict[int,int]
            Dictionary holding mapping from global to local element IDs for this proc
        """
        # Do sorting on root proc
        if self.comm.rank == 0:
            # List containing which poc every element belongs to
            elemPartition = self.creator.getElementPartition()
            # Create an empty list that we will use to sort what elements are on what procs
            allOwnedElementIDs = [[] for proc_i in range(self.comm.size)]
            for elemGlobalID in range(len(elemPartition)):
                ownerProc = elemPartition[elemGlobalID]
                allOwnedElementIDs[ownerProc].append(elemGlobalID)
        else:
            allOwnedElementIDs = None

        # Scatter the list from the root so each proc knows what element ID it owns
        ownedElementIDs = self.comm.scatter(allOwnedElementIDs, root=0)
        # Create dictionary that gives the corresponding local ID for each global ID owned by this proc
        globalToLocalElementIDDict = {
            gID: lID for lID, gID in enumerate(ownedElementIDs)
        }

        return globalToLocalElementIDDict

    def getElementObject(self, componentID, objectIndex):
        """
        Return TACS element object corresponding to component ID and object index.

        Parameters
        ----------
        componentID : int
            Component ID number

        objectIndex : int
            Object index number.
            Some components may have multiple TACS element types associated with them.
            This index specifies which object should be returned.

        Returns
        -------
        elemObj : tacs.TACS.Element
            TACS element object.
        """
        pointer = self.elemObjectNumByComp[componentID][objectIndex]
        return self.elemObjects[pointer]

    def setElementObject(self, componentID, objectIndex, elemObject):
        """
        Set TACS element object to component ID and object index.

        Parameters
        ----------
        componentID : int
            Component ID number

        objectIndex : int
            Object index number.
            Some components may have multiple TACS element types associated with them.
            This index specifies which object should be assigned.

        elemObject : tacs.TACS.Element
            TACS element object.
        """
        pointer = self.elemObjectNumByComp[componentID][objectIndex]
        self.elemObjects[pointer] = elemObject

    def getElementObjectForElemID(self, elemID, nastranOrdering=False):
        """
        Return TACS element object corresponding to specified element ID.

        Parameters
        ----------
        elemID : int
            Element ID number

        nastranOrdering : bool
            Flag signaling whether elemIDs are in TACS (default) or NASTRAN (grid IDs in bdf file) ordering
            Defaults to False.

        Returns
        -------
        elemObj : tacs.TACS.Element
            TACS element object.
        """
        # Convert to tacs numbering, if necessary
        if nastranOrdering:
            elemID = self.idMap(elemID, self.nastranToTACSElemIDDict)
        # Get the pointer for the tacs element object for this element
        elemObjNum = self.elemObjectNumByElem[elemID]
        elemObj = self.elemObjects[elemObjNum]
        return elemObj

    def createTACSAssembler(self, varsPerNode, massDVs):
        """
        Setup TACSCreator object responsible for creating TACSAssembler

        Parameters
        ----------
        varsPerNode : int
            Number of variables per node for the model.

        massDVs : dict
            Dictionary holding dv info for point masses.
        """
        self.creator = tacs.TACS.Creator(self.comm, varsPerNode)

        # Append RBE elements to element list, these are not setup by the user
        for rbe in self.bdfInfo.rigid_elements.values():
            if rbe.type == "RBE2":
                self._addTACSRBE2(rbe, varsPerNode)
            elif rbe.type == "RBE3":
                self._addTACSRBE3(rbe, varsPerNode)
            else:
                raise NotImplementedError(
                    f"Rigid element of type '{rbe.type}' is not supported"
                )

        # Append point mass elements to element list, these are not setup by the user
        for massInfo in self.bdfInfo.masses.values():
            # Find the dv dict for this mass element, if not found return empty
            dvDict = massDVs.get(massInfo.eid, {})
            self._addTACSMassElement(massInfo, varsPerNode, dvDict)

        # Check for any nodes that aren't attached to at least one element
        self._unattachedNodeCheck()

        # Setup element connectivity and boundary condition info on root processor
        if self.comm.rank == 0:
            # Set connectivity for all elements
            ptr = np.array(self.elemConnectivityPointer, dtype=np.intc)
            # Flatten nested connectivity list to single list
            conn = it.chain.from_iterable(self.elemConnectivity)
            conn = np.array([*conn], dtype=np.intc)
            objectNums = np.array(self.elemObjectNumByElem, dtype=np.intc)
            self.creator.setGlobalConnectivity(
                self.bdfInfo.nnodes, ptr, conn, objectNums
            )

            # Set up the boundary conditions
            bcDict = {}
            for spc_id in self.bdfInfo.spcs:
                for spc in self.bdfInfo.spcs[spc_id]:
                    # Loop through every node specifed in this spc and record bc info
                    for j, nastranNode in enumerate(spc.nodes):
                        # If constrained node doesn't exist in bdf
                        if nastranNode not in self.bdfInfo.node_ids:
                            self._TACSWarning(
                                f"Node ID {nastranNode} (Nastran ordering) is referenced by an SPC,  "
                                "but the node was not defined in the BDF file. Skipping SPC."
                            )
                            continue

                        # Convert to TACS node ID
                        tacsNode = self.idMap(nastranNode, self.nastranToTACSNodeIDDict)

                        # If node hasn't been added to bc dict yet, add it
                        if tacsNode not in bcDict:
                            bcDict[tacsNode] = {}

                        # Loop through each dof and record bc info if it is included in this spc
                        for dof in range(varsPerNode):
                            # Add 1 to get nastran dof number
                            nastranDOF = dof + 1
                            if spc.type == "SPC":
                                # each node may have its own dofs uniquely constrained
                                constrainedDOFs = spc.components[j]
                                # The boundary condition may be forced to a non-zero value
                                constrainedVal = spc.enforced[j]
                            else:  # SPC1?
                                # All nodes always have the same dofs constrained
                                constrainedDOFs = spc.components
                                # This boundary condition is always 0
                                constrainedVal = 0.0
                            # if nastran dof is in spc components string, add it to the bc dict
                            if self._isDOFInString(constrainedDOFs, nastranDOF):
                                bcDict[tacsNode][dof] = constrainedVal

            # Convert bc information from dict to list
            bcnodes = []
            bcdofs = []
            bcptr = [0]
            bcvals = []
            numbcs = 0
            for tacsNode in bcDict:
                bcnodes.append(tacsNode)
                # Store constrained dofs for this node
                dofs = bcDict[tacsNode].keys()
                bcdofs.extend(dofs)
                # Store enforced bc value
                vals = bcDict[tacsNode].values()
                bcvals.extend(vals)
                # Increment bc pointer with how many constraints weve added for this node
                numbcs += len(bcDict[tacsNode])
                bcptr.append(numbcs)

            # Recast lists as numpy arrays
            bcnodes = np.array(bcnodes, dtype=np.intc)
            bcdofs = np.array(bcdofs, dtype=np.intc)
            bcptr = np.array(bcptr, dtype=np.intc)
            bcvals = np.array(bcvals, dtype=self.dtype)
            # Set boundary conditions in tacs
            self.creator.setBoundaryConditions(bcnodes, bcptr, bcdofs, bcvals)

            # Set node locations
            Xpts = self.bdfInfo.get_xyz_in_coord(fdtype=self.dtype, sort_ids=False)
            self.creator.setNodes(Xpts.flatten())

        # Set the elements for each component
        self.creator.setElements(self.elemObjects)

        self.assembler = self.creator.createTACS()

        # errored here
        self.globalToLocalElementIDDict = self.getGlobalToLocalElementIDDict()

        # If any multiplier nodes were added, record their local processor indices
        localIDs = self.getLocalNodeIDsFromGlobal(
            self.multiplierNodeIDs, nastranOrdering=False
        )
        self.ownedMultiplierNodeIDs = [localID for localID in localIDs if localID >= 0]

        return self.assembler

    def _isDOFInString(self, constrained_dofs, dof):
        """
        Determine if dof number (nastran numbering) occurs in constraint string.

        Parameters
        ----------
        constrained_dofs : str
            String containing list of dofs (ex. '123456')

        dof : int or str
            nastran dof number to check for

        Returns
        -------
        found : bool
            Flag indicating if specified dof is in string.
        """
        # Convert to string, if necessary
        if isinstance(dof, int):
            dof = "%d" % dof
        # pyNastran only supports 0,1,2,3,4,5,6 as valid dof components
        # For this reason, we'll treat 0 as if its 7, since it's traditionally never used in nastran
        if dof == "7":
            dof = "0"
        location = constrained_dofs.find(dof)
        # if dof is found, return true
        if location > -1:
            return True
        else:
            return False

    def _addTACSRBE2(self, rbeInfo, varsPerNode):
        """
        Method to automatically set up RBE2 element from bdf file for user.
        User should *NOT* set these up in their elemCallBack function.

        Parameters
        ----------
        rbeInfo : pyNastran.bdf.cards.elements.rigid.RBE2
            pyNastran object holding rbe info.

        varsPerNode : int
            Number of variables per node for the model.
        """
        indepNode = rbeInfo.independent_nodes
        depNodes = []
        depConstrainedDOFs = []
        dummyNodes = []
        dofsAsString = rbeInfo.cm
        dofsAsList = self.dofStringToList(dofsAsString, varsPerNode)
        for node in rbeInfo.dependent_nodes:
            depNodes.append(node)
            depConstrainedDOFs.extend(dofsAsList)
            # add dummy nodes for all lagrange multiplier
            dummyNodeNum = (
                max(self.bdfInfo.node_ids) + 1
            )  # Next available nastran node number
            # Add the dummy node coincident to the dependent node in x,y,z
            self.bdfInfo.add_grid(dummyNodeNum, self.bdfInfo.nodes[node].xyz)
            # Update Nastran to TACS ID mapping dicts, since we just added new nodes to model
            self.nastranToTACSNodeIDDict[dummyNodeNum] = self.bdfInfo.nnodes - 1
            dummyNodes.append(dummyNodeNum)

        conn = indepNode + depNodes + dummyNodes
        nTotalNodes = len(conn)
        # Add dummy nodes to lagrange multiplier node list
        self.numMultiplierNodes += len(dummyNodes)
        tacsIDs = self.idMap(dummyNodes, self.nastranToTACSNodeIDDict)
        self.multiplierNodeIDs.extend(tacsIDs)
        # Append RBE information to the end of the element lists
        self.elemConnectivity.append(self.idMap(conn, self.nastranToTACSNodeIDDict))
        self.elemConnectivityPointer.append(
            self.elemConnectivityPointer[-1] + nTotalNodes
        )
        rbeObj = tacs.elements.RBE2(
            nTotalNodes, np.array(depConstrainedDOFs, dtype=np.intc)
        )
        self.elemObjectNumByElem.append(len(self.elemObjects))
        self.elemObjects.append(rbeObj)
        return

    def _addTACSRBE3(self, rbeInfo, varsPerNode):
        """
        Method to automatically set up RBE3 element from bdf file for user.
        User should *NOT* set these up in their elemCallBack function.

        Parameters
        ----------
        rbeInfo : pyNastran.bdf.cards.elements.rigid.RBE3
            pyNastran object holding rbe info.

        varsPerNode : int
            Number of variables per node for the model.
        """
        depNode = rbeInfo.dependent_nodes
        depConstrainedDOFs = self.dofStringToList(rbeInfo.refc, varsPerNode)

        # add dummy node for lagrange multipliers
        dummyNodeNum = max(self.bdfInfo.node_ids) + 1  # Next available node number
        # Add the dummy node coincident to the dependent node in x,y,z
        self.bdfInfo.add_grid(dummyNodeNum, self.bdfInfo.nodes[depNode[0]].xyz)
        # Update Nastran to TACS ID mapping dicts, since we just added new nodes to model
        self.nastranToTACSNodeIDDict[dummyNodeNum] = self.bdfInfo.nnodes - 1
        dummyNodes = [dummyNodeNum]
        # Add dummy node to lagrange multiplier node list
        self.numMultiplierNodes += len(dummyNodes)
        tacsIDs = self.idMap(dummyNodes, self.nastranToTACSNodeIDDict)
        self.multiplierNodeIDs.extend(tacsIDs)

        # Get node and rbe3 weight info
        indepNodes = []
        indepWeights = []
        indepConstrainedDOFs = []
        for depNodeGroup in rbeInfo.wt_cg_groups:
            wt = depNodeGroup[0]
            dofsAsString = depNodeGroup[1]
            dofsAsList = self.dofStringToList(dofsAsString, varsPerNode)
            for node in depNodeGroup[2]:
                indepNodes.append(node)
                indepWeights.append(wt)
                indepConstrainedDOFs.extend(dofsAsList)

        conn = depNode + indepNodes + dummyNodes
        nTotalNodes = len(conn)
        # Append RBE information to the end of the element lists
        self.elemConnectivity.append(self.idMap(conn, self.nastranToTACSNodeIDDict))
        self.elemConnectivityPointer.append(
            self.elemConnectivityPointer[-1] + nTotalNodes
        )
        rbeObj = tacs.elements.RBE3(
            nTotalNodes,
            np.array(depConstrainedDOFs, dtype=np.intc),
            np.array(indepWeights),
            np.array(indepConstrainedDOFs, dtype=np.intc),
        )
        self.elemObjectNumByElem.append(len(self.elemObjects))
        self.elemObjects.append(rbeObj)
        return

    def _addTACSMassElement(self, massInfo, varsPerNode, dvDict):
        """
        Method to automatically set up TACS mass elements from bdf file for user.
        User should *NOT* set these up in their elemCallBack function.

        Parameters
        ----------
        massInfo : pyNastran.bdf.cards.elements.mass.PointMassElement
            pyNastran object holding rbe info.

        varsPerNode : int
            Number of variables per node for the model.

        dvDict : dict
            Dictionary holding dv info for point mass.
        """
        if massInfo.type == "CONM2":
            m = massInfo.mass
            I11, I12, I22, I13, I23, I33 = massInfo.I
            # Create dict with input args for PointMassConstitutive
            massArgs = {
                "m": m,
                "I11": I11,
                "I22": I22,
                "I33": I33,
                "I12": I12,
                "I13": I13,
                "I23": I23,
            }
            # Update mass arguments with user-defined dv info
            massArgs.update(dvDict)
            con = tacs.constitutive.PointMassConstitutive(**massArgs)
        elif massInfo.type == "CONM1":
            M = np.zeros(21)
            M[0:6] = massInfo.mass_matrix[0:, 0]
            M[6:11] = massInfo.mass_matrix[1:, 1]
            M[11:15] = massInfo.mass_matrix[2:, 2]
            M[15:18] = massInfo.mass_matrix[3:, 3]
            M[18:20] = massInfo.mass_matrix[4:, 4]
            M[20] = massInfo.mass_matrix[5, 5]
            # off-diagonal moment of inertia terms have to be negated, since they aren't in nastran convention
            M[16] *= -1.0
            M[17] *= -1.0
            M[19] *= -1.0
            con = tacs.constitutive.GeneralMassConstitutive(M=M)
        else:
            raise NotImplementedError(
                f"Mass element of type '{massInfo.type}' is not supported"
            )

        # Append point mass information to the end of the element lists
        conn = [massInfo.node_ids[0]]
        self.elemConnectivity.append(self.idMap(conn, self.nastranToTACSNodeIDDict))
        self.elemConnectivityPointer.append(self.elemConnectivityPointer[-1] + 1)
        # Create tacs object for mass element
        massObj = tacs.elements.MassElement(con)
        self.elemObjectNumByElem.append(len(self.elemObjects))
        self.elemObjects.append(massObj)
        return

    def _unattachedNodeCheck(self):
        """
        Check for any nodes that aren't attached to an element.
        Notify the user and throw an error if we find any.
        This must be checked before creating the TACS assembler or a SegFault may occur.
        """
        numUnattached = 0
        if self.comm.rank == 0:
            # Flatten connectivity to a single list
            flattenedConn = it.chain.from_iterable(self.elemConnectivity)
            # uniqueify and order all element-attached nodes
            attachedNodes = set(flattenedConn)
            # Loop through each node in the bdf and check if it's in the element node set
            for nastranNodeID in self.bdfInfo.node_ids:
                tacsNodeID = self.idMap(nastranNodeID, self.nastranToTACSNodeIDDict)
                if tacsNodeID not in attachedNodes:
                    if numUnattached < 100:
                        self._TACSWarning(
                            f"Node ID {nastranNodeID} (Nastran ordering) is not attached to any element in the model. "
                            f"Please remove this node from the mesh and try again."
                        )
                    numUnattached += 1

        # Broadcast number of found unattached nodes
        numUnattached = self.comm.bcast(numUnattached, root=0)
        # Raise an error if any unattached nodes were found
        if numUnattached > 0:
            raise self._TACSError(
                f"{numUnattached} unattached node(s) were detected in model. "
                f"Please make sure that all nodes are attached to at least one element."
            )

    def dofStringToList(self, dofString, numDOFs):
        """
        Converts a dof string to a boolean list.
        Examples:
            '123' -> [1, 1, 1, 0, 0, 0]
            '1346' -> [1, 0, 1, 1, 0, 1]

        Parameters
        ----------
        dofString : string
            String containing list of dofs (ex. '123456')

        numDOFs : int
            Number of dofs in model

        Returns
        -------
        dofList : list[int]
            List of booleans indicating which dofs are present in input string.
        """
        dofList = []
        for dof in range(numDOFs):
            dof = str(dof + 1)
            loc = dofString.find(dof)
            if loc > -1:
                dofList.append(1)
            else:
                dofList.append(0)
        return dofList

    def idMap(self, fromIDs, tacsIDDict):
        """
        Translate fromIDs numbering from nastran numbering to tacs numbering.
        If node ID doesn't exist in nastranIDList, return -1 for entry

        Parameters
        ----------
        fromIDs : int or list[int]
            IDs in Nastran numbering

        tacsIDDict : dict[int, int]
            ID mapping dict generated by `_updateNastranToTACSDicts`

        Returns
        -------
        toIDs : int or list[int]
            IDs in TACS numbering
        """
        # Input is a list return a list
        if hasattr(fromIDs, "__iter__"):
            toIDs = [None] * len(fromIDs)
            # Iterate through list and call function recursively one element at a time
            for i, id in enumerate(fromIDs):
                toIDs[i] = self.idMap(id, tacsIDDict)
            return toIDs
        # Input is a int, return an int
        else:
            if fromIDs in tacsIDDict:
                return tacsIDDict[fromIDs]
            else:
                return -1

    @property
    def allLocalNodeIDs(self):
        """
        get list of tacs_ids for each nastran node owned across all processors
        nastran_node = array_idx + 1
        tacs_idx = output id
        nastran_node - 1 => tacs_idx owned by this proc
        """

        self._nastranToLocalNodeIDMap()
        all_struct_ids = None
        local_maps = self.comm.gather(self._local_map, root=0)
        if self.comm.rank == 0:
            all_struct_ids = np.zeros((self.bdfInfo.nnodes), dtype=int)
            for local_map in local_maps:
                for key in local_map:
                    all_struct_ids[int(key)] = map[int(key)]
            all_struct_ids = list(all_struct_ids)
        # broadcast to other procs
        all_struct_ids = self.comm.bcast(all_struct_ids, root=0)
        return all_struct_ids

    def _getLocalNodeIDs(self):
        """
        get the local struct ids owned by this processor, full list when comm is None
        -1 for each idx not owned by this processor
        """
        num_nodes = self.bdfInfo.nnodes
        bdf_nodes = [_ for _ in range(num_nodes)]
        return self.getLocalNodeIDsFromGlobal(bdf_nodes, nastranOrdering=False)

    def _nastranToLocalNodeIDMap(self):
        """
        write the map nastran_node - 1 => tacs_idx on each processor
        """
        local_struct_ids = self._getLocalNodeIDs()
        id_map = []
        for arr_idx, struct_id in enumerate(local_struct_ids):
            if struct_id != -1:
                id_map.append({arr_idx: struct_id})
        self._local_map = id_map
        return id_map
