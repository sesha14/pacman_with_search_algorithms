# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self,state): #initialize an actionlist for use
        self.actionList = []
        for i in range(0, 5):
            self.actionList.append(Directions.STOP)
        return
    def initial_list(self,state): # create a successors list for calculating initial gameevaluation score
        temps = state
        m=0
        while (m != 5):
            current = temps
            if (temps.isWin() == 1 or temps.isLose() == 1):  # do not proceed if the current state is win or lose state
                temps = current
                break
            else:
                temps = temps.generatePacmanSuccessor(self.actionList[m])
                if (temps is None): #check for none states in successors
                    temps = current
                    break
            m += 1
        return temps
    def random_list(self,state,next_sequence): #create an actionlist with 50% randomness
        actions = state.getAllPossibleActions()
        for l in range(0, len(next_sequence)):
            if (random.randint(0, 1) == 0):
                next_sequence[l] = actions[random.randint(0, len(actions) - 1)]
        return next_sequence
    def check(self,generated,initial): #check if the generated score is local maxima or not
        if(generated>=initial):
            return 1
        else:
            return 0

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        k=0 #to keep checking none state
        actions = state.getAllPossibleActions()
        if(state.isWin()):#check if curent state is winstate and stop
            return Directions.STOP

        temps=self.initial_list(state)#create a successors list for calculating initial gameevaluation score
        initial_score = gameEvaluation(state,temps) #initial game score calculation

        for i in range(0, len(self.actionList)):
            self.actionList[i] = actions[random.randint(0, len(actions) - 1)]

        #create an actionlist where each action has 50% chance to be random
        next_sequence = self.actionList
        while (k!=1): #continue until none state occurs
            next_sequence1 = self.random_list(state,next_sequence)
            temps = state
            for n in range(0, len(next_sequence1)):
                if (temps.isWin()==1 or temps.isLose()==1):
                        break
                else:
                    current = temps
                    q=next_sequence1[n]
                    temps = temps.generatePacmanSuccessor(q)
                    if (temps is None):
                        k = 1
                        temps = current
                        break
            #calculate game score of 50% random list successors
            generated_score= gameEvaluation(state,temps)
            pt=self.check(generated_score,initial_score)
            if (pt==1):
                self.actionList = next_sequence
                initial_score = generated_score

            if(k!=1):#assign only if successors are not None
                next_sequence = self.actionList

        action_tobetaken = self.actionList[0]
        return action_tobetaken

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    pop_count = 8 #chromosomes count
    chromo_count = 5 #chromosome length
    def registerInitialState(self, state):
        self.actionList = []
        for i in range(0, 5):
            self.actionList.append(Directions.STOP)
        return

    def tobetaken(self,after_ranking,Final): #return population with ranking and score after rank selection
        s=len(after_ranking)-1
        score = after_ranking[s][1]
        if Final[1]<=score :
            tobetaken = [after_ranking[s][0][0], score]
            return tobetaken
        return Final

    def cross_over(self,nextgen,Parents): #perform crossover
        random1 = random.randint(0, 10)
        children={
            "chromofirst":[],"chromosecond":[]
        }
        if random1 > 7:
            nextgen.append(Parents[0])
            nextgen.append(Parents[1])
        else:
            for m in range(0, len(Parents[0])):
                random2 = random.randint(0, 1)
                if random2 is 1:
                    children["chromofirst"].append(Parents[0][m])
                    children["chromosecond"].append(Parents[1][m])
                else:
                    children["chromofirst"].append(Parents[1][m])
                    children["chromosecond"].append(Parents[0][m])
            nextgen.append(children["chromofirst"])
            nextgen.append(children["chromosecond"])
        return nextgen

    def mutation(self,state,nextgen): #perform mutation
        actions = state.getAllPossibleActions()
        k=0
        while(k!=len(nextgen)):
            random1 = random.randint(0, 100)
            if random1 > 10:
                continue
            else:
                random2 = random.randint(0, 8)
                if(random2 in range(0,len(nextgen[k])-1)):
                    q=random.randint(0, len(actions) - 1)
                    nextgen[k][random2] = actions[q]
            k+=1
        return nextgen

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        Actions_tobetaken=[]
        n=0
        first_pop = []
        Final = ("", 0)

        #first generation population
        for i in range(0, self.pop_count):#population=8
            for j in range(0, self.chromo_count):#each chromosome of length 5
                available = state.getAllPossibleActions()
                k=random.randint(0, len(available) - 1)
                self.actionList[j] = available[k]
            first_pop.append(self.actionList)

        ranks=[]
        while n!=1 :#do not proceed if none is returned
            new_pop = []
            nextgen=[]
            c=0
            while(c!=self.pop_count):
                j=0
                curr = state
                while j in range(0, self.chromo_count):
                    if (curr.isWin()==0 and curr.isLose() == 0):
                        prev = curr
                        curr = curr.generatePacmanSuccessor(first_pop[c][j])
                        if curr is None:
                            n=1
                            curr = prev
                            break
                    else:
                        break
                    j+=1
                c += 1
                eval_value = gameEvaluation(state,curr)
                ranks.append(eval_value)
            for l in range(0,self.pop_count):
                new_pop.append((first_pop[l],ranks[l]))

            #allot ranks to population generated
            def key1(x):
                return x[1]
            after_ranking = sorted(new_pop, key=key1)
            Actions_tobetaken=self.tobetaken(after_ranking,Final)

            ## Selecting pair using rank selection for crossover
            i=0
            while(i!=5):
                Parents = []
                parent1 = random.randint(1, 36)
                if parent1 in range(1,8):
                    r1= 7
                elif parent1 in range(9,15):
                    r1= 6
                elif parent1 in range(16,21):
                    r1= 5
                elif parent1 in range(22,26):
                    r1= 4
                elif parent1 in range(27,30):
                    r1= 3
                elif parent1 in range(31,33):
                    r1= 2
                elif parent1 in range(34,35):
                    r1= 1
                else:
                    r1= 0
                Parents.append(after_ranking[r1][0])
                parent2 = random.randint(1, 36)
                if parent2 in range(1, 8):
                    r2 = 7
                elif parent2 in range(9, 15):
                    r2 = 6
                elif parent2 in range(16, 21):
                    r2 = 5
                elif parent2 in range(22, 26):
                    r2 = 4
                elif parent2 in range(27, 30):
                    r2 = 3
                elif parent2 in range(31, 33):
                    r2 = 2
                elif parent2 in range(34, 35):
                    r2 = 1
                else:
                    r2 = 0
                Parents.append(after_ranking[r2][0])
                #perform crossover
                nextgen = self.cross_over(nextgen,Parents)
                i+=1
            #perform mutation
            first_pop=self.mutation(state,nextgen)
        return Actions_tobetaken[0]

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    f= False
    def registerInitialState(self, state):
        return
    def calculate_selection(self,initial): #Calculate UCT formula values
        selected = {"element":[],"score":0}
        for ch in initial.children:
            Xi=(ch.evaluation / ch.visited_count)
            N=initial.visited_count
            ni=ch.visited_count
            UCT_formula = Xi+ 1* math.sqrt(2*math.log(N)/ni)
            if UCT_formula > selected["score"]:
                selected["score"] = UCT_formula
                selected["element"] = ch
            elif UCT_formula == selected["score"]:
                selected["element"].append(ch)
        initial = random.choice(selected["element"])
        return initial

    def general_treeconstruction(self, initial): #building the tree
        while initial is not None :
            if(initial.win & initial.lose ==0):
                while initial.visited==0: #we must create the children of taken node
                    actions = initial.legalActions
                    if len(actions)==0:
                        initial.visited=1
                    actions = actions[random.randint(0, len(actions) - 1)]
                    consecu_state = initial.state.generatePacmanSuccessor(actions)
                    if consecu_state is None:
                        self.f = True
                        return None
                    else:
                        new = newchild_struct(consecu_state, initial, actions)
                        initial.children.append(new)
                    if initial.visited==1:
                        return new

                if initial.win & initial.lose !=0:  #we can now select the node to be expanded
                    initial=self.calculate_selection(initial)

    def Simulation(self, ele): #do simulation for selected element
        state1 = ele.state
        m=0
        while(m!=5 and state1 is not None):#perform upto 5 rollouts
            m+=1
            gh = state1.getLegalPacmanActions()
            new = state1.generatePacmanSuccessor(random.choice(gh))
            if (state1.win & state1.lose ==0) & (new is not None):
                gh=state1.getLegalPacmanActions()
                new = state1.generatePacmanSuccessor(random.choice(gh))
                break
            else:
                state1 = new
        return gameEvaluation(self.rootnode, state1)

    def mostvisited_nodes(self,root): #to return highest number of visits nodes
        highest_visited=[]
        max_count=0
        for k in root.children:#find the count of max visits
            if(k.visited_count>max_count):
                max_count=k.visited_count
            else:
                continue
        for l in root.children:#find the element with highest visit count
            if l.visited_count==max_count:
                highest_visited.append(l)
            else:
                continue
        return highest_visited



    # GetAction Function: Called with every frame
    def getAction(self, state):
        self.f = False
        self.rootnode=state
        initial_parent = newchild_struct(state, None, None)
        rollout_outcome=-2
        while not self.f:
            child1 = self.general_treeconstruction(initial_parent) #tree construction
            if child1 is not None: #send the game evaluation score till root
                rollout_outcome = self.Simulation(child1) #perform 5 rollouts
            if(rollout_outcome!=-2):
                ch=child1.parent
                while(ch is not None): #do backpropagation
                    child1.game_score += rollout_outcome
                    child1.visited_count += 1
                    ch = child1.parent
            else:
                break

        #return mostvisited state action
        mostvisited_nodes=self.mostvisited_nodes(initial_parent)
        l=random.choice(mostvisited_nodes).action
        return l
class newchild_struct: #structure of each tree node
    createchild=[]
    par=[]
    count=0
    def __init__(self,state, parent, curr_action):
        self.state = state
        self.legalActions = state.getLegalPacmanActions()
        self.win = state.isWin()
        self.lose = state.isLose()
        self.parent = self.par
        self.children = self.createchild
        self.visited_count = self.count
        self.game_score = self.count
        self.visited = self.count
        self.action = curr_action


# fractal dimension (“coastline approximation” - 1)
# The mean, standard error, and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features
class KNN:
    def __init__(self, k):
        #KNN state here
        #Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        print(featureA)
        diffs = (featureA - featureB)**2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        self.X_train=X
        self.y_train=y
        #training logic here
        #input is an array of features and labels

    def get_neighbors(self, X, Xtest, neighbor_count):
        distance_list=[]
        neighbors=[]
        for t in X:
            d = self.distance(t, Xtest)
            distance_list.append((d, t))

        def key1(x):
            return x[1]
        list = sorted(distance_list, key=key1)

        for i in range(neighbor_count):
            neighbors.append(list[i][1])
        return neighbors

    def predict(self, X):
        k_neighbors=self.get_neighbors(self.X_train, self.y_train, self.k)
        print(k_neighbors)
        #Run model here
        #Return array of predictions where there is one prediction for each set of features
        return None