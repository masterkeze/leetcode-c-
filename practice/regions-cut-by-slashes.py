class Solution:
    def regionsBySlashes(self, grid):
        cnt = 1
        return 1
    def regionsBySlashes2(self, grid):
        if len(grid) == 0:
            return 0
        if len(grid[0]) == 0:
            return 0
        cnt = 1
        length = len(grid[0])
        concernsNow = [["\\","/"]] * length
        concernsNext = [[]] * length
        for j in range(len(grid)):
            row = grid[j]
            for i in range(length):
                char = row[i]
                if not(char in concernsNow[i]) and i != 0 and i != length - 1:
                    continue
                if char == "\\":
                    if i == length - 1:
                        cnt += 1
                    else:
                        if j == len(grid) - 1:
                            cnt += 1
                        elif row[i+1] == "/" and (i == length - 2 or row[i+1] in concernsNow[i+1]):
                            cnt += 1
                        concernsNext[i+1].append("\\")
                    concernsNext[i].append("/")

                if char == "/":
                    if i == 0:
                        cnt += 1
                    else:
                        if j == len(grid) - 1:
                            cnt += 1
                        concernsNext[i-1].append("/")
                    concernsNext[i].append("\\")
            #print(concernsNow,concernsNext)
            print(concernsNow)
            print(concernsNext)
            print()
            concernsNow = concernsNext
            concernsNext = [[]] * length
        return cnt


s = Solution()
#print(s.regionsBySlashes(["/\\","\\/"]))
print(s.regionsBySlashes(["\\/\\ "," /\\/"," \\/ ","/ / "]))
