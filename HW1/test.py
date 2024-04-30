import sys

class TreeNode:
    def __init__(self, name, freq, parent):
        self.name = name
        self.count = freq
        self.nodeLink = None
        self.parent = parent
        self.children = {}
    
    def Increment(self, count):
        self.count += count

def LoadData(InputFile):
    with open(InputFile, 'r') as file:
        data = file.read()
        transactions = data.strip().split('\n')
        transactions = [list(map(int, transaction.split(','))) for transaction in transactions]
    return transactions

def CreateInitSet(data):
    retDict = {}
    for trans in data:
        retDict[frozenset(trans)] = 1
    return retDict

def CreateFPTree(data, min_support):
    HeaderTable = {}
    
    for transaction in data:
        for item in transaction:
            HeaderTable[item] = HeaderTable.get(item,0) + data[transaction]
    for k in list(HeaderTable):
        if HeaderTable[k] < float(min_support):
            del(HeaderTable[k])
            
    frequent_itemset = set(HeaderTable.keys())

    if len(frequent_itemset) == 0:
        return None, None

    for k in HeaderTable:
        HeaderTable[k] = [HeaderTable[k], None]

    retTree = TreeNode('Null Set', 1, None)
    for itemset,count in data.items():
        frequent_transaction = {}
        for item in itemset:
            if item in frequent_itemset:
                frequent_transaction[item] = HeaderTable[item][0]
        if len(frequent_transaction) > 0:
            #to get ordered itemsets form transactions
            ordered_itemset = [v[0] for v in sorted(frequent_transaction.items(), key=lambda p: p[1], reverse=True)]
            #to update the FPTree
            UpdateTree(ordered_itemset, retTree, HeaderTable, count)
    return retTree, HeaderTable
            
def UpdateTree(itemset, FPTree, HeaderTable, count):
    if itemset[0] in FPTree.children:
        FPTree.children[itemset[0]].Increment(count)
    else:
        FPTree.children[itemset[0]] = TreeNode(itemset[0], count, FPTree)
        if HeaderTable[itemset[0]][1] == None:
            HeaderTable[itemset[0]][1] = FPTree.children[itemset[0]]
        else:
            UpdateNodeLink(HeaderTable[itemset[0]][1], FPTree.children[itemset[0]])
    if len(itemset) > 1:
        UpdateTree(itemset[1:], FPTree.children[itemset[0]], HeaderTable, count)
        
def UpdateNodeLink(Test_Node, Target_Node):
    while (Test_Node.nodeLink != None):
        Test_Node = Test_Node.nodeLink

    Test_Node.nodeLink = Target_Node
    
def FPTreeUptransveral(leaf_Node, prefixPath):
    if leaf_Node.parent != None:
        prefixPath.append(leaf_Node.name)
        FPTreeUptransveral(leaf_Node.parent, prefixPath)
        
def FindPrefixPath(basePat, TreeNode):
    Conditional_patterns_base = {}

    while TreeNode != None:
        prefixPath = []
        FPTreeUptransveral(TreeNode, prefixPath)
        if len(prefixPath) > 1:
            Conditional_patterns_base[frozenset(prefixPath[1:])] = TreeNode.count
        TreeNode = TreeNode.nodeLink

    return Conditional_patterns_base

def MineTree(FPTree, HeaderTable, minSupport, prefix, frequent_itemset):
    bigL = [v[0] for v in sorted(HeaderTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        new_frequentset = prefix.copy()
        new_frequentset.add(basePat)
        
        freq_itemset[frozenset(new_frequentset)] = HeaderTable[basePat][0]

        Conditional_pattern_bases = FindPrefixPath(basePat, HeaderTable[basePat][1])
        
        Conditional_FPTree, Conditional_header = CreateFPTree(Conditional_pattern_bases, minSupport)

        if Conditional_header != None:
            MineTree(Conditional_FPTree, HeaderTable, minSupport, new_frequentset, frequent_itemset)

def OutputResult(freq_pat, output_name):
    with open(output_name, 'w') as f:
        for pattern in freq_pat:
            support = freq_pat[pattern]
            f.write(f"{','.join(map(str, pattern))}:{support:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Input Error: python3 你的學號_hw1.py [min support] [輸入檔名] [輸出檔名]")
        sys.exit(1)
    min_support = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    transactions =  LoadData(input_file)
    total_transactions = len(transactions)
    
    InitSet = CreateInitSet(transactions)
    
    FPtree, HeaderTable = CreateFPTree(InitSet, min_support)
    
    freq_itemset = {}
    MineTree(FPtree, HeaderTable, min_support, set([]), freq_itemset)
    
    OutputResult(freq_itemset, output_file)
