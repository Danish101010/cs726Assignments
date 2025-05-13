import json

from copy import deepcopy

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################
class DSU:
    def __init__(self,n):
        self.size = [1 for _ in range(n)]
        self.parent = [i for i in range(n)]

    def find(self,i):
        if i == self.parent[i]:
            return i
        
        return self.find(self.parent[i])
    def unionbysize(self,i,j):
        irep = self.find(i)
        jrep = self.find(j)
        if irep == jrep:
            return 
        isize = self.size[irep]
        jsize = self.size[jrep]
        if(isize<jsize):
            self.parent[irep]=jrep
            self.size[jrep]+= self.size[irep]
            
        else:
            self.parent[jrep] = irep
            self.size[irep] += self.size[jrep]

        
def is_simplicial(vertex, adj_list):
    ## A helper function which checks if a vertex is simplicial or not
    if vertex not in adj_list:
        return False
    neighbors = list(adj_list[vertex])
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] not in adj_list[neighbors[i]]:
                return False
    return True

def min_deg(adj_list, remaining):
    ## A helper function which returns the vertex with minimum degree on induced graph
    mn_deg = float('inf')
    min_ver = None
    
    for vertex in remaining:
        degree = len(adj_list[vertex])
        if degree < mn_deg:
            mn_deg = degree
            min_ver = vertex
            
    return min_ver

def make_simplicial(vertex, adj_list):
    ## If the vertex is not simplicial, this function makes it simplicial by adding required edges
    edges = []
    neighbors = list(adj_list[vertex])
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] not in adj_list[neighbors[i]]:
                edges.append((neighbors[i], neighbors[j]))
    
    return edges

def add_edges(edges, adj_list):
    ## A helper function which adds edges to the graph (both induced and triangulated)
    for u, v in edges:
        if v not in adj_list[u]:
            adj_list[u].append(v)
        if u not in adj_list[v]:
            adj_list[v].append(u)

def remove_vertex(vertex , adj_list):
    ## A helper function which removes a vertex from the graph
    neigh = adj_list[vertex]

    for x in neigh:
        adj_list[x].remove(vertex)

    adj_list[vertex] =[]


#Note we have take this code from gfg  https://www.geeksforgeeks.org/maximal-clique-problem-recursive-solution/
def bron_kerbosch(R, P, X, graph):
    if not P and not X:
        yield R
    while P:
        v = P.pop()
        yield from bron_kerbosch(
            R.union({v}),
            P.intersection(graph[v]),
            X.intersection(graph[v]),
            graph
        )
        X.add(v)


def get_maximal_cliques(adj):
    ## A helper function which returns the maximal cliques in a graph given adjacency list of triangulated graph 


    graph =  {i: list(adj[i]) for i in range(len(adj))}
    
    graph = {key: set(graph[key]) for key in graph}
    all_cliques = list(bron_kerbosch(set(), set(graph.keys()), set(), graph))
    if all_cliques:
        max_clique_size = max(len(clique) for clique in all_cliques)
    else:
        max_clique_size = -1
    # print(max_clique_size)
    

    return all_cliques



def mstfind(V,edges):
    ## A helper function which returns the minimum spanning tree of a graph given the number of vertices and edges

    edges.sort(key = lambda x :x[2])
    dsu = DSU(V)
    mst = []
    cost = 0

    for u, v,w in edges:  
        if dsu.find(u) != dsu.find(v):  
            dsu.unionbysize(u, v)
            mst.append((u, v, w))
            cost += w
            if len(mst) == V - 1: 
                break

    return mst, cost

def imap(maxim_clique, givenClique):
    
    max_vars = sorted(list(maxim_clique))
    orig_vars = sorted(list(givenClique))
    
    positions = []
    for i in orig_vars:
        if i in max_vars:
            id  = max_vars.index(i) 
            positions.append(id)

    mapping = {}


    for i in range(2**len(orig_vars)):
        l = []
        for j in range(2**len(max_vars)):
            i_bits = [(i >> k) & 1 for k in range(len(orig_vars) - 1, -1, -1)]
            j_bits = [(j >> k) & 1 for k in range(len(max_vars) - 1, -1, -1)]
            if all(j_bits[pos] == i_bits[idx] for idx, pos in enumerate(positions)):
                l.append(j)
        mapping[i] = l 


        
    return mapping



def messages_init(j_tree,maxim_clique):
    ## initializes all the message from every node to its neighbours 
    return { (i, j): [1]*2**(abs(len(set(maxim_clique[i]) & set(maxim_clique[j])))) for (i,_) in enumerate(j_tree) for j in j_tree[i] }

def sendMessage(x, y, pot, messages, max_cliques, j_tree):

    
    c = set(sorted(max_cliques[x]))
    c_d  = set(sorted(max_cliques[y]))
    ## Abhi Psic * mul term kr rha hu 
    newmsg = pot[x][:]
    # messages[(x,y)] = [1]*2**(len(c & c_d))

    common =  sorted(list(c & c_d))
    # sumvars =  sorted(list(c - set(common)))
    # d = set(j_tree[x]) -set(y)

    for i in j_tree[x]:
        if i== y:
            continue
        msg = messages[(i,x)]
        common2 = sorted(set((max_cliques[i])) & set(sorted(max_cliques[x])))
        # c = {1, 2, 3}
        # common2 = {2, 3}
        mp = imap(c,common2)
        # print(c, common2)
        # print(mp)
        # exit(1)
        for t,v in mp.items():
            for j in v:
                newmsg[j] *= msg[t]

    
    ans_msg = [0]*2**len(common)
    mp2 = imap(c, common)
    # print(mp2)
    for k,v in mp2.items():
        for i in v:
            ans_msg[k] += newmsg[i]

    messages[(x,y)] = ans_msg
    


    # print(f"Message sent from {x} to {y}: {new_msg}")


def message_passing(j_tree,pot,maxim_clique):

    ## this function implement message passing algorithm 
    ## firstly i inialitize the messages  as dictionary whose keys are (i,j) where i is the sender and j is the receiver
    ## and values are the messages sent from i to j and it is list(table ) of size 2^(common elements of i and j)
    messages = messages_init(j_tree,maxim_clique)
    receivedSet = {i: set() for i in range(len(j_tree))} ## this is the set which keeps track of the neighbours from which the message is received

    nodes = len(j_tree)
    it = 0  

    while it < 10:
        for i in range(nodes):

            neigh = set(j_tree[i])
            diff = neigh - receivedSet[i]  
            
            if len(diff) > 1: ## if node i has not reaceived enough messages then it cant send message as well 
                ## so check other nodes 
                continue
            elif len(diff) == 1: ## if node i accepted all messages from its neighboours excepts one then it can compute the message and send that to that neighbour
                x = diff.pop()  
                receivedSet[x].add(i) 
                sendMessage(i, x,pot,messages,maxim_clique,j_tree)  ## this function computes the message and sends it to the neighbour x
            else: ## if all messages are received then compute the message and send it to all neighbours
                for v in neigh:
                    receivedSet[v].add(i)
                    sendMessage(i,v,pot,messages,maxim_clique,j_tree)
        
        it += 1 
        # print(messages) 

    return messages

def imap2(x, y):
    ## it is helper function which return what indices in original clique marginal corresponds to compressed table of 
    ## given variable at index 0 and 1
   
    mp = {0: [], 1: []}
    idx = sorted(list(y)).index(x)
    # print("enetered imap2")
    # print(x, y, idx)
    for i in range(2**len(y)):
         ## i am using the fact that in bit representation of i if correspond variable (x ) position is 
         ## 1 then map it to 1st position in marginal pprobability of variable
        if (i >> (len(y) - 1 - idx)) & 1:
            mp[1].append(i)
        else:
            mp[0].append(i)
    # print("exited imap2")
    return mp


def marg_xi(potential ,clique_vars, xi ):
    l = [0]*2
    
    ## computes the marginal probability of variable xi given it is contained in clique_vars

    # print("entered function")
    # print(potential)
    # print(clique_vars, xi)
    mp = imap2(xi,clique_vars)
    # print(y ,clique_vars , mp)
    # print(potential)
    # print(mp)
    for k,v in mp.items():
        for j in v:
            l[k] += potential[j]
    # print(l)
    # exit(1)
    # print(l)
    return l



class item:
    def __init__(self, assignment={}, prob=0):
        self.assignment = assignment
        self.prob = prob


def topk_init(j_tree,maxim_clique):
    return { (i, j): [item()]*2**(abs(len(set(maxim_clique[i]) & set(maxim_clique[j])))) for (i,_) in enumerate(j_tree) for j in j_tree[i] }


def topKsendMessage(x, y, pot, messages, max_cliques, j_tree, k_value = 2):

    c = set(sorted(max_cliques[x]))
    c_d  = set(sorted(max_cliques[y]))
    ## Abhi Psic * mul term kr rha hu 
    # print(pot[x][:])
    # print("MESSAGE", x, y)
    newmsg = [[] for _ in range(2**len(c))]
    for j in range(2**len(c)):
        newmsg[j].append(item())
        newmsg[j][0].assignment = {}
        newmsg[j][0].prob = 0
    # print(newmsg)
    # for i in range(2**len(c)):
    #     print(newmsg[i][0].assignment)
    
    for i in range(2**len(c)):
        newmsg[i][0].prob = pot[x][i]

        msg = [int(alpha) for alpha in bin(i)[2:].zfill(len(max_cliques[x]))]
        clique = sorted(max_cliques[x])
        for k in range(len(clique)):
            # print(clique[k], msg[k])
            newmsg[i][0].assignment[clique[k]] = msg[k]
        # print(newmsg[i][0].assignment)

    common =  sorted(list(c & c_d))

    for i in j_tree[x]:
        if i == y:
            continue
        msg = messages[(i,x)]
        common2 = sorted(set((max_cliques[i])) & set(sorted(max_cliques[x])))
        mp = imap(c,common2)
        for t,v in mp.items():
            for j in v:
                replace_msg = []
                
                for entry1 in newmsg[j]:
                    for entry2 in msg[t]:
                        for k in entry1.assignment:
                            if k in entry2.assignment:
                                if entry1.assignment[k] != entry2.assignment[k]:
                                    print("Shit")
                                    assert(False)
                            
                        new_entry = item()
                        new_entry.prob = entry1.prob * entry2.prob
                        new_entry.assignment = entry1.assignment.copy()
                        new_entry.assignment.update(entry2.assignment)
                        replace_msg.append(new_entry)
                
                newmsg[j] = replace_msg
                
    
    ans_msg = [[] for _ in range(int(2**len(common)))]
    
    mp2 = imap(c, common)
    
    for k,v in mp2.items():
        for i in v:
            ans_msg[k].extend(newmsg[i])
    
    for j in range(len(ans_msg)):
        ans_msg[j].sort(key = lambda x: x.prob, reverse = True)
        ans_msg[j] = ans_msg[j][:k_value]
    
    messages[(x,y)] = ans_msg

def topk_message_passing(j_tree,pot,maxim_clique, k):
    messages = topk_init(j_tree,maxim_clique)
    receivedSet = {i: set() for i in range(len(j_tree))}
    nodes = len(j_tree)
    it = 0  

    while it < 10:
        for i in range(nodes):

            neigh = set(j_tree[i])
            diff = neigh - receivedSet[i]  
            
            if len(diff) > 1:
                continue
            elif len(diff) == 1:
                x = diff.pop()  
                receivedSet[x].add(i)
                topKsendMessage(i, x, pot, messages,maxim_clique,j_tree, k)  
            else:
                for v in neigh:
                    receivedSet[v].add(i)
                    topKsendMessage(i,v, pot, messages,maxim_clique,j_tree, k)
        
        it += 1 

    # for k in messages:
    #     print(k)
    #     for topk in messages[k]:
    #         print('Top k:', end=' ')
    #         for j in topk:
    #             print('[', j.assignment, ',', j.prob, ']', end='')
    #         print()
    return messages


class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.data = data 
        self.VariablesCount = self.data['VariablesCount']
        self.Potentials_count = self.data['Potentials_count']
        self.CliqueNPotentials = self.data['Cliques and Potentials']
        self.k = self.data['k value (in top k)']
        self.cliques = []
        self.potentials = []
    
        self.adj_list = [[] for _ in range(self.VariablesCount)]
        for item in self.CliqueNPotentials:
            neigh = list(item['cliques'])
            for i in range(len(neigh)):
                for j in range(i+1,len(neigh)):
                    if(neigh[j] not in self.adj_list[neigh[i]]):
                        self.adj_list[neigh[i]].append(neigh[j])
                    if(neigh[i] not in self.adj_list[neigh[j]]):
                        self.adj_list[neigh[j]].append(neigh[i])

            self.cliques.append(set(item['cliques']))
            self.potentials.append(item['potentials'])


        


    

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        ## Here I have implemented the algorithm for triangulation given  in the slides 

        self.triangulated = deepcopy(self.adj_list)  ## This is the adjacency list of the graaph which will be triangulated finally 

        induced = deepcopy(self.adj_list)  ## This is the induced graph since we have to perform minimum degree and simplicial calculations on this graph

        vertices = list(range(len(self.adj_list)))
        rem = set(vertices)   ## This is the set of vertices which are not yet triangulated
        self.ordering = []
        

        while rem:

            sim_ver = None
            for v in rem:
                
                if is_simplicial(v, induced): ## We need to check simplicial only on induced graph
                    sim_ver = v
                    break

            if sim_ver is None: ## If no simplicial vertex is found, we need to find the vertex with minimum degree on induced graph
                vertex = min_deg(induced, rem)
            else:
                vertex = sim_ver
                
            
            
            if not is_simplicial(vertex, self.triangulated): ## If the vertex is not simplicial, we need to make it simplicial
                new_edges = make_simplicial(vertex, induced) ## it returns the edges which need to make it simplicial on induced graph
                add_edges(new_edges, self.triangulated) ## then add eges on both induced and triangulated graph
                add_edges(new_edges,induced)

            rem.remove(vertex)
            self.ordering.append(vertex)
            remove_vertex(vertex,induced)

        ## We can obtain maximal cliques by passing the triangulated graph to the helper function get_maximal_cliques defined above 
        self.maxim_cliques = get_maximal_cliques(self.triangulated)
                

 


    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        ##now next step is to get thr junction tree from the maximal cliques
        ## we will use the maximal cliques to create the junction tree 
        ## the idea is to create edges between the maximal cliques which have non empty intersection and the weight is the cardinality of the intersection
        
        edge_list = [] 
        self.n = len(self.maxim_cliques)
        for ind1 in range(len(self.maxim_cliques)):
            for ind2 in range(ind1 + 1, len(self.maxim_cliques)):  
                wt = len(set(self.maxim_cliques[ind1]) & set(self.maxim_cliques[ind2]))  ## weight is number of common elements
                if wt > 0:  
                    edge_list.append((ind1,ind2,-wt))
        # lst = [[] for _ in range(len(self.maxim_cliques))]


        # for u, v, w in edge_list:
        #     lst[u].append(v)
        #     lst[v].append(u)  

    
        # visualize(lst)
        # print(lst)  
        

        ## we passed this edge list to the helper function mstfind which returns the minimum spanning tree of the graph 
        ## actually the algorithm will return maximum spanning tree (which is what we want ) since the weights are negative
        mst,_ = mstfind(len(self.maxim_cliques),edge_list)   ## this step creates the maximum spanning tree adjacency list
        self.mst_adj = [[] for _ in range(self.n)]
        for u , v , w in mst:
            self.mst_adj[u].append(v)
            self.mst_adj[v].append(u)

        # print(self.maxim_cliques)
        # print(self.mst_adj)

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.

        """
        ## Now we need to assign potentials to the cliques in the junction tree
        ## we will use the given potentials to assign them to the cliques

         
         ## i have initialized the junction potentials with 1 every element in this list is a table which will contain potential for a junction clique using original clique
        self.junction_potentials = [[1 for _ in range(2**len(self.maxim_cliques[i]))] for i in range(len(self.maxim_cliques))]

        # self.junction_clique_members = dict()
        # print(self.cliques)


        ## the idea here is to create tables for each junction clique using the original cliques , i have used index mapping 
        ## to map the original cliques potential indices to the junction cliques indices 
        for i, x in enumerate(self.cliques):
            l = []
            ind = -1
            mx = float('inf')
            ## for every clique given find the junction clique which contains it and has minimum size i think it works even if not minimum size 
            for j, y in enumerate(self.maxim_cliques):

                if x.issubset(y) and len(y) < mx: 
                    l = y
                    ind = j
                    mx = len(y)

            if ind == -1:
                continue
            potential_x = self.potentials[i] ## the potential of the original clique
            # print(i, x, l)
            index_mapping = imap(l,x)  ## the  imap function takes the original clique and junction clique and returns the mapping of indices of potential of original clique to potential of junction clique
            
            # if tuple(l) in self.junction_clique_members:
            #     self.junction_clique_members[tuple(l)].append(x)
            # else:
            #     self.junction_clique_members[tuple(l)] = [x]
            

            ## now just replicate the potentials based on assignment in the junction tree table
            for pot_idx, value in enumerate(potential_x):
                clique_idx = index_mapping[pot_idx] 
                for k in clique_idx:
                    self.junction_potentials[ind][k] *= value  

        # print(self.junction_clique_members)

        # print(self.junction_potentials)


        ## at this step we assigned potential to cliques in juntion tree
        

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """

        ## getting the partition function using message passing algorithm
        self.messages = message_passing(self.mst_adj,self.junction_potentials,self.maxim_cliques) ## explained its working in the original function
        ## self.message is the diction where keys are (a,b) where a is the sender and b is the receiver and value is the message sent from a to b (message is the table whose size is 2^(common elements of a and b))


        self.clique_marginals = {i : self.junction_potentials[i].copy() for i, _ in enumerate(self.maxim_cliques)} ## this step creates the clique marginals which are initialized with the junction potentials
        ## as mentioned in slides P(xc) = Psi(Xc )* Mul(messages from all neighbours to c)

        # print(self.junction_potentials)
        for c in range(len(self.maxim_cliques)):
            for d in self.mst_adj[c]:
                cliq = self.maxim_cliques[c]
                a = set(self.maxim_cliques[d])
                b = set(self.maxim_cliques[c])
                ## for every message from a neighbour of c 
                ## we first map the indices of commmon variables to inndices of max clique c
                ## then we multiply the message with the clique marginal of c
                common = sorted(a & b)
                mp = imap(cliq,common)
                msg = self.messages[(d,c)].copy()

                for k,v in mp.items():
                    for l in v:
                        self.clique_marginals[c][l] *= msg[k]



        #at this line we get the clique marginals
        

        # print(self.junction_potentials)
        # print(self.clique_marginals)
        # print(self.maxim_cliques)
        #for z_value we simply sum the table values of any clique marginals table i have take 0th table
        self.z_value = sum(self.clique_marginals[0])
        return self.z_value
        






    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        

        ## once we got the clique marginal computing the maarginal prob wrt any variable is easy
        ## find the clique which contains that variable the compree the table using mapping of indices 
        self.marginals = []

        for i in range(self.VariablesCount):
            ans = []
            for j, l in enumerate(self.maxim_cliques):
            
                if i in l:

                    # print(j,l,i)

                    ans = marg_xi(self.clique_marginals[j],l,i)  ## this function returns the (non normalized actually) marginal probability of variable i given it is contained in clique j
                    ans = [x/self.z_value for x in ans]   ##normalize it here 
                    
                    # print(ans)

                    
            self.marginals.append(ans)

        return self.marginals

            



    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """

        ## here i used the similar idea for computing top k assignments as i used for computing marginals
        ## the only difference is that i need to keep track of top k assignments for each junction clique
        ## not compreesing (summing up ) the table values of clique marginals wrt index mapping
       
        self.topkmessages = topk_message_passing(self.mst_adj,self.junction_potentials,self.maxim_cliques, self.k)
        max_cliques = self.maxim_cliques
        pot = self.junction_potentials
        j_tree = self.mst_adj
        k_value = self.k
        x = 0
        
        c = set(sorted(max_cliques[x]))
       

        newmsg = [[] for _ in range(2**len(c))]
        for j in range(2**len(c)):
            newmsg[j].append(item())
            newmsg[j][0].assignment = {}
            newmsg[j][0].prob = 0
        
        for i in range(2**len(c)):
            newmsg[i][0].prob = pot[x][i]

            msg = [int(alpha) for alpha in bin(i)[2:].zfill(len(max_cliques[x]))]
            clique = sorted(max_cliques[x])
            for k in range(len(clique)):
                newmsg[i][0].assignment[clique[k]] = msg[k]

        common =  []

        for i in j_tree[x]:
            msg = self.topkmessages[(i,x)]
            common2 = sorted(set((max_cliques[i])) & set(sorted(max_cliques[x])))
            mp = imap(c,common2)
            for t,v in mp.items():
                for j in v:
                    replace_msg = []
                    
                    for entry1 in newmsg[j]:
                        for entry2 in msg[t]:
                            for k in entry1.assignment:
                                if k in entry2.assignment:
                                    if entry1.assignment[k] != entry2.assignment[k]:
                                        print("Shit")
                                        assert(False)
                                
                            new_entry = item()
                            new_entry.prob = entry1.prob * entry2.prob
                            new_entry.assignment = entry1.assignment.copy()
                            new_entry.assignment.update(entry2.assignment)
                            replace_msg.append(new_entry)
                    
                    newmsg[j] = replace_msg
                    
        
        ans_msg = [[] for _ in range(int(2**len(common)))]
        
        mp2 = imap(c, common)
        
        for k,v in mp2.items():
            for i in v:
                ans_msg[k].extend(newmsg[i])
        
        for j in range(len(ans_msg)):
            ans_msg[j].sort(key = lambda x: x.prob, reverse = True)
            ans_msg[j] = ans_msg[j][:k_value]
        
        l1 = []

        for i in range(k_value):
            dict1 = {}
            sorted_assignment_1 = list(dict(sorted(ans_msg[0][i].assignment.items())).values())
            dict1['assignment'] = sorted_assignment_1
            dict1['probability'] = ans_msg[0][i].prob/self.z_value
            l1.append(dict1)
        
        return l1
        



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Sample_Testcase[1].json')
    evaluator.get_output()
    evaluator.write_output('TestCases_Output.json')