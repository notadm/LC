from typing import List, Tuple

def containsDuplicate(nums: List[int]) -> bool:
    h = {}
    for n in nums:
        if n in h: return True
        h[n] = 1
    return False

containsDuplicate("ada") # True
containsDuplicate("abc") # False

def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t): return False
    countS, countT = {}, {}
    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[t[i]] = 1 + countT.get(t[i], 0)
    return countS == countT

isAnagram("Anagram", "mnAgraa") # True
isAnagram("ab", "ad") # False

def getConcatenation(nums: List[int]) -> List[int]:
    ans = []
    for i in range(2):
        for n in nums: ans.append(n)
    return ans

getConcatenation([1,2,3]) # [1,2,3,1,2,3]

def replaceElements(arr: List[int]) -> List[int]:
    rightMax = -1
    for i in range(len(arr) -1, -1, -1):
        newMax = max(rightMax, arr[i])
        arr[i] = rightMax
        rightMax = newMax
    return arr

getConcatenation([17,18,5,4,6,1]) # [18,6,6,6,1,-1]

def isSubsequence(s: str, t: str) -> bool:
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]: i += 1
        j += 1
    return i == len(s)

isSubsequence("abc","afbdc") # True
isSubsequence("ace","afedc") # False

def lengthOfLastWord(s: str) -> int:
    return len(s.split()[-1])

lengthOfLastWord("h ello   world  ") # 5

def twoSum(nums: List[int], target: int) -> List[int]:
    M = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in M: return [M[diff], i]
        M[n] = i

twoSum([2,7,11,15], 9) # [0,1]
twoSum([3,2,4], 6) # [1,2]

def longestCommonPrefix(strs: List[str]) -> str:
    res = ""
    for i in range(len(strs[0])):
        for s in strs:
            if i == len(s) or s[i] != strs[0][i]: return res
        res += strs[0][i]
    return res

longestCommonPrefix(["flower","flow","flight"]) # "fl"
longestCommonPrefix(["dog","racecar","car"]) # ""

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    ans = {} 
    for s in strs:
        count = [0] * 26
        for c in s: count[ord(c) - ord("a")] += 1
        if tuple(count) in ans: ans[tuple(count)].append(s)
        else: ans[tuple(count)] = [s]
    return list(ans.values())

groupAnagrams(["eat","tea","tan","ate","nat","bat"]) 
# [["bat"],["nat","tan"],["ate","eat","tea"]]

def pascalTriangle(n:int):
    ans = []
    for i in range(n):
        ans.append([1] * (i+1))
        for j in range(1,i): ans[-1][j] = ans[-2][j-1] + ans[-2][j] 
    return ans

pascalTriangle(5) 
# [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

def removeElement(nums: List[int], val: int) -> int:
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k

removeElement([3,2,2,3], 3) # [2,2,_,_]

def numUniqueEmails(emails: list[str]) -> int:
    unique_emails = {}
    for email in emails:
        local_name, domain_name = email.split('@')
        local_name = local_name.split('+')[0]
        local_name = local_name.replace('.','')
        email = local_name + '@' + domain_name
        unique_emails[email] = 1
    return len(unique_emails)

numUniqueEmails(["a@x.com","b@x.com","c@x.com"]) # 3

def isIsomorphic(s: str, t: str) -> bool:
    S = {}
    for i,j in zip(s,t):
        if i in S and S[i] != j: return False
        S[i] = j
    return True

isIsomorphic("egg", "add") # True
isIsomorphic("foo", "bar") # False

def canPlaceFlowers(x: List[int], n: int) -> bool:
    x.insert(0,-1)
    x.append(-1)
    for i in range(1,len(x)-1):
        if x[i] == 1: x[i-1],x[i+1] = -1,-1
    k = 0
    for i in x: 
        if (i == 0): k += 1
    return n <= k

canPlaceFlowers([0,0,1,0,0,0,1], 2) # True
canPlaceFlowers([0,0,1,0,0,0,1], 3) # False

def majorityElement(nums: List[int]) -> int:
    res, count = 0, 0
    for n in nums:
        if count == 0: res = n
        count += (1 if n == res else -1)

majorityElement([2,2,1,1,1,2,2]) # 2

def nextGreaterElement(x:List[int], y:List[int]) -> List[int]:
    S, ans = {}, [-1] * len(x)
    for i in range(len(y)): S[y[i]] = i
    for i in range(len(x)): 
        j = S[x[i]]
        for k in range(j, len(y)):
            if (x[i] < y[k]): 
                ans[i] = y[k]
                break
    return ans

nextGreaterElement([4,1,2], [1,3,4,2])  # [-1,3,-1]
nextGreaterElement([2,4], [1,2,3,4]) # [3,-1]

def pivotIndex(nums: List[int]) -> int:
    l,total = 0, sum(nums)
    for i in range(len(nums)):
        r = total - nums[i] - l 
        if l == r: return i
        l += nums[i]
    return -1

pivotIndex([1,7,3,6,5,6]) # 3

class NumArray:
    def __init__(self, nums: List[int]):
        cur, self.prefix = 0, []
        for n in nums:
            cur += n
            self.prefix.append(cur)
    def sumRange(self, left: int, right: int) -> int:
        r = self.prefix[right] 
        l = self.prefix[left - 1] if left > 0 else 0
        return r - l

numArray = NumArray([-2,0,3,-5,2,-1])
numArray.sumRange(0, 2) # 1
numArray.sumRange(2, 5) # -1 
numArray.sumRange(0, 5) # -3

def findDisappearedNumbers(nums: List[int]) -> List[int]:
    for n in nums: nums[abs(n)-1] = -1 * abs(nums[abs(n)-1])
    res = []
    for i, n in enumerate(nums):
        if n > 0: res.append(i + 1)
    return res

findDisappearedNumbers([4,3,2,7,8,2,3,1]) # [5,6]

def maxNumberOfBalloons(text: str) -> int:
    B = "balloon"
    K,A,S = {},{},{}
    for x in B: S[x] = S.get(x,0) + 1
    for x in B: K[x] = 1 / S[x]
    for x in B: A[x] = 0
    for x in text: 
        if x in A: A[x] = round(A[x] + K[x])
    return min(list(A.values())) 

maxNumberOfBalloons("loonbalxballpoon") # 2

def wordPattern(pattern: str, s: str) -> bool:
    s = s.split()
    H = {}
    for i in range(len(s)):
        if pattern[i] in H and H[pattern[i]] != s[i]: return False
        else: H[pattern[i]] = s[i]
    return True

wordPattern("abba", "dog cat cat dog") # True
wordPattern("abba", "dog cat cat bee") # False

def sortArray(n: List[int]) -> List[int]:
    if (len(n) == 1): return n
    l,r,s = sortArray(n[:len(n)//2]), sortArray(n[len(n)//2:]), []
    i = j = 0
    def add(x:list,v:int):
        s.append(x[v])
        return v + 1
    def merge(x:list,v:int):
        while(v<len(x)): v = add(x,v)
        return s
    while(i<len(l) and j<len(r)):
        if l[i]<r[j]: i = add(l,i)
        else: j = add(r,j)
    return merge(l,i) if i<len(l) else merge(r,j)

sortArray([-2,3,5,1,1,2,-1,13,8]) # [-2,-1, 1, 1, 2, 3, 5, 8, 13]

def topKFrequent(nums: List[int], k: int) -> List[int]:
    count, freq, r = {}, [[] for i in range(len(nums) + 1)], []
    for n in nums: count[n] = 1 + count.get(n, 0)
    for n, c in count.items(): freq[c].append(n)
    for i in range(len(freq)-1, 0, -1):
        for j in freq[i]:
            r.append(j)
            if (len(r) == k): return r;
    return r 

topKFrequent([1,1,1,2,2,3], 2) # [1,2]
topKFrequent([1], 1) # [1]


