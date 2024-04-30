''' Apriori Algorithms'''
import sys

def Preprocess(input_file):
    raw_transactions = []
    transactions = []
    max_trans_len = 1
    with open(input_file, 'r') as file:
        data = file.read()
        raw_transactions = data.strip().split('\n')
        for item in raw_transactions:
            item_list = item.split(',')
            max_trans_len = max(max_trans_len, len(item_list))
            transactions.append(item_list)
    return max_trans_len, transactions

def CreateC1(data):
    c1 = set()
    for trans in data:
        for item in trans:
            ItemSet = frozenset([item])
            c1.add(ItemSet)
    return c1

def Generate(data, c, min_support, support_data):
    lk = set()
    count = {}
    for transactions in data:
        for item in c:
            if item.issubset(transactions):
                if item not in count:
                    count[item] = 1
                else:
                    count[item] += 1
    for item in count:
        if count[item]>=min_support:
            lk.add(item)
            support_data[item] = count[item]
    return lk

def apriori (items, previous):
    for item in items:
        sub = items - frozenset([item])
        if sub not in previous:
            return False
    
    return True

def CreateCk(previous, k):
    ck = set()
    PreviousLen = len(previous)
    PreviousList = list(previous)
    for i in range(PreviousLen):
        for j in range(i+1, PreviousLen):
            L1 = list(PreviousList[i])
            L2 = list(PreviousList[j])
            L1.sort()
            L2.sort()
            if L1[:k-2] == L2[:k-2]:
                items = PreviousList[i] | PreviousList[j]
                if apriori(items, previous):
                    ck.add(items)
    return ck

def AprioriImplement(data, max_trans_len, min_support):
    c1 = CreateC1(data)
    support_data = {}
    # Generate Lk by Ck
    l1 = Generate(data, c1, min_support, support_data)
    # Previous Lk
    previous = l1.copy()
    for i in range(2, max_trans_len+1):
        ci = CreateCk(previous, i)
        li = Generate(data, ci, min_support, support_data)
        if len(li) == 0:
            break
        previous = li.copy()
    return support_data

def SortFreqPat(freq_pat):
    new_type_of_freq = []
    for i in freq_pat:
        new_list = []
        num = []
        sup_val = freq_pat[i]

        for j in i:
            num.append(j)
        num.sort()

        new_list.append(num)
        new_list.append(sup_val)
        new_type_of_freq.append(new_list)

    new_type_of_freq.sort()
    return new_type_of_freq

def Output(output_file, freq_pat):
    with open(output_file, 'w') as file:
        for pattern in freq_pat:
            pattern_str = ','.join(pattern[0])
            support_str = "{:.4f}".format(pattern[1] / len(transactions))
            file.write(pattern_str + ':' + support_str + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Input Error: python3 你的學號_hw1.py [min support] [輸入檔名] [輸出檔名]")
        sys.exit(1)
    min_support = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    max_trans_len, transactions = Preprocess(input_file)
    
    # convert min_support from ratio to count
    if (min_support < 1):
        min_support = int(min_support * len(transactions))  
    
    freq_pat = AprioriImplement(transactions, max_trans_len, min_support)
    
    sorted_freq_pat = SortFreqPat(freq_pat)
    
    Output(output_file, sorted_freq_pat)