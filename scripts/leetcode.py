import bisect


class Solution:
    def numMatchingSubseq(self, S, words):
        """
        :type S: str
        :type words: List[str]
        :rtype: int
        """
        char_index = {}
        for i, s in enumerate(S):
            char_index.setdefault(s, []).append(i)
        cnt = 0
        for w in words:
            if self.helper(char_index, w):
                cnt += 1
        return cnt

    def helper(self, index, word):
        old_cur = -1
        for w in word:
            cur = self.search(index, w, old_cur)
            if cur == -1:
                return False
            old_cur = cur
        return True

    def search(self, index, c, old_cur):
        if c not in index:
            return -1
        ind = bisect.bisect_right(index[c], old_cur)
        if ind >= len(index[c]):
            return -1
        return index[c][ind]

    def search_(self, index, c, old_cur):
        if c not in index:
            return -1
        inds = index[c]
        begin = 0
        end = len(inds) - 1
        while begin <= end:
            mid = (begin + end) // 2
            if inds[mid] <= old_cur:
                begin += 1
            else:
                end -= 1
        if begin >= len(inds):
            return -1
        return inds[begin]


if __name__ == "__main__":
    #a = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    #a = [[".",".","9","7","4","8",".",".","."],["7",".",".",".",".",".",".",".","."],[".","2",".","1",".","9",".",".","."],[".",".","7",".",".",".","2","4","."],[".","6","4",".","1",".","5","9","."],[".","9","8",".",".",".","3",".","."],[".",".",".","8",".","3",".","2","."],[".",".",".",".",".",".",".",".","6"],[".",".",".","2","7","5","9",".","."]]
    b = Solution().numMatchingSubseq("abcde",["a","bb","acd","ace"])
    print(b)