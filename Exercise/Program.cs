using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Exercise
{
    public class Fancy
    {
        int _tick;
        List<Node> nodes;
        List<Operation> operations;
        public struct Node {
            public long val;
            public int inserted;
        }
        public struct Operation {
            public int type;
            public int parameter;
        }
        public Fancy()
        {
            _tick = 0;
            nodes = new List<Node>();
            operations = new List<Operation>();
        }

        public void Append(int val)
        {
            Node node = new Node() { val = val, inserted = _tick };
            nodes.Add(node);
        }

        public void AddALL(int inc)
        {
            Operation operation = new Operation() { type = 1, parameter = inc };
            operations.Add(operation);
            _tick += 1;
        }

        public void MultAll(int m)
        {
            Operation operation = new Operation() { type = 2, parameter = m };
            operations.Add(operation);
            _tick += 1;
        }

        public int GetIndex(int idx)
        {
            if (idx >= nodes.Count)
            {
                return -1;
            }

            Node node = nodes[idx];
            int inserted = node.inserted;
            long val = node.val;
            long output = val;
            for (int i = inserted; i < operations.Count; i++)
            {
                Operation operation = operations[i];
                if (operation.type == 1)
                {
                    output += operation.parameter;
                }
                else
                {
                    output *= operation.parameter;
                }
                if (output > Int32.MaxValue)
                {
                    output %= 1000000007;
                }
            }
            output %= 1000000007;
            node.inserted = _tick;
            node.val = output;
            return (int)(output);
        }
    }
    class Solution
    {
        public double FindMaxAverage(int[] nums, int k)
        {
            
            long current = nums.Take(k).ToList().Sum();
            long sum = current;
            Console.WriteLine(current);
            for (int i = 1; i + k - 1 < nums.Length; i++)
            {
                current = current - nums[i - 1] + nums[i + k - 1];
                if (current > sum)
                {
                    sum = current;
                }
                Console.WriteLine(current);
            }
            
            return (double)sum / k;
        }

        public double[] MedianSlidingWindow(int[] nums, int k)
        {
            double[] output = new double[nums.Length - k + 1];
            if (k == 1)
            {
                for (int i = 0; i < nums.Length; i++)
                {
                    output[i] = (double)nums[i];
                }
                return output;
            }

            for (int i = 0; i < output.Length; i++)
            {
                List<int> cur = nums.Skip(i).Take(k).ToList();
                cur.Sort();
                if (k%2 == 0)
                {
                    output[i] = ((long)cur[k / 2 - 1] + cur[k / 2]) / 2;
                }
                else
                {
                    output[i] = (double)cur[(k - 1) / 2];
                }
            }

            return output;
        }
        public int[] FairCandySwap(int[] A, int[] B)
        {
            int[] output = new int[2];
            int sumA = 0;
            int sumB = 0;
            for (int i = 0; i < A.Length; i++)
            {
                sumA += A[i];
            }
            for (int i = 0; i < B.Length; i++)
            {
                sumB += B[i];
            }
            List<int> AList = new List<int>(A);
            List<int> BList = new List<int>(B);
            AList.Sort();
            BList.Sort();
            int j = 0;
            int k = 0;
            while (j < A.Length && k < B.Length)
            {
                int ACandy = AList[j];
                int BCandy = BList[k];
                if (sumA - ACandy + BCandy == sumB - BCandy + ACandy)
                {
                    output[0] = ACandy;
                    output[1] = BCandy;
                    return output;
                }else if (sumA - ACandy + BCandy > sumB - BCandy + ACandy)
                {
                    j++;
                }
                else
                {
                    k++;
                }
            }

            return output;
        }
        public struct Node
        {
            int row;
            int column;
            int cost;
        }
        public enum Direction
        {
            Left, Right, Up, Down
        }
        public int GetCostTo(int[][] heights, int row, int column, Direction direction)
        {
            int rows = heights.Length;
            int columns = heights[0].Length;
            if (row < 0 || row >= rows || column < 0 || column >= columns)
            {
                return -1;
            }
            if (row == 0 && direction == Direction.Up)
            {
                return -1;
            }
            if (row == rows - 1 && direction == Direction.Down)
            {
                return -1;
            }
            if (column == 0 && direction == Direction.Left)
            {
                return -1;
            }
            if (column == columns - 1 && direction == Direction.Right)
            {
                return -1;
            }
            int output = -1;
            switch (direction)
            {
                case Direction.Left:
                    output = Math.Abs(heights[row][column] - heights[row][column - 1]);
                    break;
                case Direction.Right:
                    output = Math.Abs(heights[row][column] - heights[row][column + 1]);
                    break;
                case Direction.Up:
                    output = Math.Abs(heights[row][column] - heights[row - 1][column]);
                    break;
                case Direction.Down:
                    output = Math.Abs(heights[row][column] - heights[row + 1][column]);
                    break;
                default:
                    break;
            }
            return output;
        }
        public int MinimumEffortPath(int[][] heights)
        {
            // dijkstra gogogo
            int rows = heights.Length;
            int columns = heights[0].Length;

            return 0;
        }
        public int PivotIndex(int[] nums)
        {
            int length = nums.Length;
            if (length == 0)
            {
                return -1;
            }
            if (length == 1)
            {
                return 0;
            }
            int leftSum = 0;
            int rightSum = 0;
            for (int i = 1; i < length; i++)
            {
                rightSum += nums[i];
            }
            for (int i = 0; i < length - 1; i++)
            {
                if (leftSum == rightSum)
                {
                    return i;
                }
                else
                {
                    leftSum += nums[i];
                    rightSum -= nums[i + 1];
                }
            }

            return leftSum == rightSum ? length - 1 : -1;
        }
        public int PivotIndex2(int[] nums)
        {
            int length = nums.Length;
            if (length == 0)
            {
                return -1;
            }
            if (length == 1)
            {
                return 0;
            }
            int leftPointer = -1;
            int leftSum = 0;
            int rightPointer = length;
            int rightSum = 0;

            int output = -1;
            while (true)
            {
                int diff = Math.Abs(rightSum - leftSum);
                if (rightPointer - leftPointer == 2)
                {
                    if (diff == 0)
                    {
                        output = leftPointer + 1;
                    }
                    break;
                }

                int leftExpectedDiff = Math.Abs(rightSum - (leftSum + nums[leftPointer + 1]));
                int rightExpectedDiff = Math.Abs(rightSum + nums[rightPointer - 1] - leftSum);

                if (leftExpectedDiff <= diff && rightExpectedDiff > diff)
                {
                    leftPointer += 1;
                    leftSum += nums[leftPointer];
                }
                else
                {
                    rightPointer -= 1;
                    rightSum += nums[rightPointer];
                }
                Console.WriteLine(leftPointer.ToString() + ":" + leftSum.ToString());
                Console.WriteLine(rightPointer.ToString() + ":" + rightSum.ToString());
            }

            return output;
        }
        public class PriorityQueue<T>
        {
            private readonly List<T> list;
            private readonly IComparer<T> comparer;

            public PriorityQueue(IComparer<T> comparer = null)
            {
                this.comparer = comparer ?? Comparer<T>.Default;
                this.list = new List<T>();
            }
            //O(Log N)
            public void Add(T x)
            {
                list.Add(x);

                var child = Count - 1;
                while (child > 0)
                { // child index; start at end
                    int parent = (child - 1) / 2;// parent index
                                                 // child item is larger than (or equal) parent so we're done
                    if (comparer.Compare(list[parent], x) <= 0) break;
                    list[child] = list[parent];
                    child = parent;
                }
                if (Count > 0) list[child] = x;
            }

            public T Peek() => list[0];
            //O(Log N)
            public T Poll()
            {
                var ret = Peek();
                var root = list[Count - 1];
                list.RemoveAt(Count - 1);
                var i = 0;//parent
                while (i * 2 + 1 < Count)
                {
                    var left = 2 * i + 1; //left child
                    if (left > Count) break;  // no children so we're done
                    var right = 2 * i + 2; // right child
                    var c = right < Count && comparer.Compare(list[right], list[left]) < 0 ? right : left;
                    if (comparer.Compare(list[c], root) >= 0) break;
                    list[i] = list[c];
                    i = c;
                }
                if (Count > 0) list[i] = root;
                return ret;
            }
            public int Count => list.Count;//Count is cached in List implelemtation

            public void DisplayHeap() => list.ForEach(x => Console.WriteLine(x));
        }
        public uint ReverseBits(uint n)
        {
            /*            var chararr = Convert.ToString(n, 2).PadLeft(32, '0').Substring(0, 32).ToCharArray();
                        Array.Reverse(chararr);
                        return Convert.ToUInt32(new string(chararr),2);*/
            string str = Convert.ToString(n, 2);
            string outstr = "";

            for (int i = 0; i < 32; i++)
            {
                outstr += str.Length - i - 1 >= 0 ? str[str.Length - i - 1] : '0';
            }

            return Convert.ToUInt32(outstr, 2);
        }
        public bool IsIsomorphic(string s, string t)
        {
            var keyset1 = new Dictionary<char, char>();
            var keyset2 = new Dictionary<char, char>();
            for (int i = 0; i < s.Length; i++)
            {
                if (!keyset1.ContainsKey(s[i]))
                {
                    keyset1.Add(s[i], t[i]);
                }
                if (!keyset2.ContainsKey(t[i]))
                {
                    keyset2.Add(t[i], s[i]);
                }
                if (keyset1[s[i]] != t[i] || keyset2[t[i]] != s[i])
                {
                    return false;
                }
            }
            return true;
        }


        public class TreeNode
        {
            public int val;
            public TreeNode left;
            public TreeNode right;
            public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
            {
                this.val = val;
                this.left = left;
                this.right = right;
            }
        }
        public struct NodeWithDepth
        {
            public TreeNode node;
            public int depth;
        }

        public int WidthOfBinaryTree(TreeNode root)
        {
            int output = 0;
            var left = new Dictionary<int, int>();
            dfs(root, 0, 0);
            void dfs(TreeNode node, int depth = 0, int pos = 0)
            {
                if (node != null)
                {
                    if (!left.ContainsKey(depth)) left[depth] = pos;
                    output = Math.Max(output, pos - left[depth] + 1);
                    dfs(node.left, depth + 1, 2 * pos);
                    dfs(node.right, depth + 1, 2 * pos + 1);
                }
            }
            return output;


            // bfs
            //var q = new Queue<NodeWithDepth>();
            //var counter = new Dictionary<int, (int,int)>();

            //NodeWithDepth nodeWithDepth = new NodeWithDepth { node = root, depth = 1 };
            //q.Enqueue(nodeWithDepth);
            //while (q.Count != 0)
            //{
            //    NodeWithDepth nodeC = q.Dequeue();
            //    TreeNode node = nodeC.node;
            //    int depth = nodeC.depth;

            //    if (!counter.ContainsKey(depth))
            //    {
            //        counter.Add(depth, (0, 0));
            //    }
            //    (int start, int length) = counter[depth];
            //    if (node != null)
            //    {
            //        q.Enqueue(new NodeWithDepth { node = node.left, depth = depth + 1 });
            //        q.Enqueue(new NodeWithDepth { node = node.right, depth = depth + 1 });
            //        int max = Math.Max(start + 1, length + 1);
            //        counter[depth] = (max, max);
            //        output = max > output ? max : output;
            //    } else
            //    {
            //        if (length != 0)
            //        {
            //            q.Enqueue(new NodeWithDepth { node = null, depth = depth + 1 });
            //            q.Enqueue(new NodeWithDepth { node = null, depth = depth + 1 });
            //            counter[depth] = (start, length + 1);
            //        }
            //    }
            //}
            //Console.WriteLine(counter.ToString());

        }
        public bool SearchMatrix(int[][] matrix, int target)
        {
            int height = matrix.Length;
            int width = matrix[0].Length;
            int top = 0;
            int bottom = height - 1;
            int mid = 0;
            while (top <= bottom)
            {
                mid = top + (bottom - top) / 2;
                if (matrix[mid][0] == target || target == matrix[mid][width-1]) return true;
                if (matrix[mid][0] < target && target < matrix[mid][width-1])
                {
                    break;
                }
                if (matrix[mid][0] < target)
                    top = mid + 1;
                else
                    bottom = mid - 1;
            }
            if (mid >= 0 && mid <= height - 1)
            {
                return BinarySearch(matrix[mid], target);
            } else
            {
                return false;
            }

        }
        public bool BinarySearch(int[] arr, int target)
        {
            int low = 0;
            int high = arr.Length - 1;
            while (low <= high)
            {
                int mid = low + (high - low) / 2;
                // Console.WriteLine($"{low},{mid},{high}");
                if (arr[mid] == target) return true;
                if (arr[mid] < target)
                    low = mid + 1;
                else
                    high = mid - 1;
            }
            return false;
        }

        public int NumDecodings(string s)
        {
            Dictionary<string, int> dict = new Dictionary<string, int>();

            for (int i = 1; i <= 26; i++)
            {
                dict.Add(i.ToString(), 1);
            }
            // index,length

            Dictionary<int, int> cache = new Dictionary<int, int>();

            int GetResult(int i)
            {
                if (i == 0) return 1;
                if (!cache.ContainsKey(i))
                {
                    cache[i] = 0;
                    if (dict.ContainsKey(s.Substring(i - 1, 1)))
                    {
                        cache[i] += GetResult(i - 1);
                    }
                    if (i >= 2 && dict.ContainsKey(s.Substring(i - 2, 2)))
                    {
                        cache[i] += GetResult(i - 2);
                    }
                    return cache[i];
                }

                return cache[i];
            }
            return GetResult(s.Length);
        }
        public int Clumsy(int N)
        {
            Dictionary<int, int> cache = new Dictionary<int, int>();
            Dictionary<int, int> cache2 = new Dictionary<int, int>();
            int[] starter = new int[] { 0, 1, 2, 6, 7, 7, 8 };
            int Multiple3(int n)
            {
                if (!cache2.ContainsKey(n))
                {
                    cache2[n] = n * (n - 1) / (n - 2);
                }
                return cache2[n];
            }
            int C(int n)
            {
                if (n <= 6)
                {
                    return starter[n];
                }
                if (!cache.ContainsKey(n))
                {
                    cache[n] = Multiple3(n) + (n - 3) - 2 * Multiple3(n - 4) + C(n - 4);
                }

                return cache[n];
            }
            // C(4) =                                     4 * 3 / 2 + 1;
            // C(8) =                     8 * 7 / 6 + 5 - 4 * 3 / 2 + 1;
            // C(12) = 12 * 11 / 10 + 9 - 8 * 7 / 6 + 5 - 4 * 3 / 2 + 1;
            // C(6) =   6 * 5 / 4 + 3 - 2 * 1 = 
            return C(N);
        }
        public bool CanDistribute(int[] nums, int[] quantity)
        {
            Dictionary<int, int> indexs = new Dictionary<int, int>();
            Dictionary<int, int> inverseIndexes = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                int num = nums[i];
                if (indexs.ContainsKey(num))
                {
                    inverseIndexes[indexs[num]] -= 1;
                    if (inverseIndexes[indexs[num]] <= 0)
                        inverseIndexes.Remove(indexs[num]);

                    indexs[num] += 1;
                } else
                {
                    indexs[num] = 1;
                }
                if (!inverseIndexes.ContainsKey(indexs[num]))
                {
                    inverseIndexes[indexs[num]] = 1;
                }
                else
                {
                    inverseIndexes[indexs[num]] += 1;
                }
            }
            List<int> sortedblocks = new List<int>(quantity);
            sortedblocks.Sort();
            Stack<int> blocks = new Stack<int>(sortedblocks);
            List<int> buckets = new List<int>();

            foreach ( var pair in inverseIndexes)
            {
                for (int i = 0; i < pair.Value; i++)
                {
                    buckets.Add(pair.Key);
                }
            }
            buckets.Sort();
            Dictionary<string, bool> cache = new Dictionary<string, bool>();

            bool visited(List<int> bkts)
            {
                string key = String.Join("", bkts);
                if (cache.ContainsKey(key))
                {
                    return true;
                }
                else
                {
                    cache[key] = true;
                    return false;
                }
            }

            bool bfs(Stack<int> blks, List<int> bkts)
            {
                if (blks.Count == 0) return true;
                if (visited(bkts)) return false;
                int blk = blks.Pop();
                for (int k = bkts.Count - 1; k >= 0; k--)
                {
                    int bkt = bkts[k];
                    if (bkt >= blk)
                    {
                        bkts[k] = bkt - blk;
                        if (bfs(blks, bkts))
                        {
                            return true;
                        }
                        else
                        {
                            bkts[k] = bkt;
                        }
                    }

                }
                blks.Push(blk);
                return false;
            }
            return bfs(blocks, buckets);
            //for (int j = ls.Count - 1; j >= 0 ; j--)
            //{
            //    int demand = ls[j];
            //    if (inverseIndexes.ContainsKey(demand) && inverseIndexes[demand] > 0)
            //    {
            //        inverseIndexes[demand] -= 1;
            //        if (inverseIndexes[demand] <= 0)
            //            inverseIndexes.Remove(demand);
            //        continue;
            //    }
            //    int maxkey = inverseIndexes.Keys.Max();
            //    if (demand <= maxkey)
            //    {
            //        inverseIndexes[maxkey] -= 1;
            //        if (inverseIndexes[maxkey] <= 0)
            //            inverseIndexes.Remove(maxkey);
            //        if (inverseIndexes.ContainsKey(maxkey - demand))
            //        {
            //            inverseIndexes[maxkey - demand] += 1;
            //        }else
            //        {
            //            inverseIndexes[maxkey - demand] = 1;
            //        }
            //    } else
            //    {
            //        return false;
            //    }
            //}

        }
        public int Trap(int[] height)
        {
            int width = height.Length;
            int blockCount = 0;
            int leftCount = 0;
            int leftHeight = 0;
            int rightCount = 0;
            int rightHeight = 0;
            for (int i = 0; i < width; i++)
            {
                blockCount += height[i];
                leftHeight = Math.Max(leftHeight, height[i]);
                rightHeight = Math.Max(rightHeight, height[width - 1 - i]);
                leftCount += leftHeight;
                rightCount += rightHeight;
            }
            int totalCount = width * leftHeight;
            // rightCount = A + B + C
            // leftCount = A + B + D
            // blockCount = B
            // totoalCount = A + B + C + D
            return (leftCount + rightCount - totalCount - blockCount) / 2;
        }

        public void flipRow(int[] row)
        {
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = 1 - row[i];
            }
        }
        public void flipCol(int[][] A, int index)
        {
            for (int i = 0; i < A.Length; i++)
            {
                A[i][index] = 1 - A[i][index];
            }
        }
        public int getScore(int[][] A)
        {
            int score = 0;
            for (int i = 0; i < A.Length; i++)
            {
                string str = string.Join("", A[i]);
                score += Convert.ToInt32(str, 2);
            }
            return score;
        }
        public int MatrixScore(int[][] A)
        {
            // 第一列都必须是1，之后每列比较下翻和不翻哪个得分高
            int width = A[0].Length;
            int height = A.Length;
            for (int c = 0; c < width; c++)
            {
                int counter = 0;
                for (int r = 0; r < height; r++)
                {
                    if (c == 0)
                    {
                        if (A[r][c] == 0)
                        {
                            flipRow(A[r]);
                        }
                    }
                    else
                    {
                        if (A[r][c] == 0)
                        {
                            counter -= 1;
                        } else
                        {
                            counter += 1;
                        }
                    }
                }
                if (counter < 0)
                {
                    flipCol(A, c);
                }
            }
            return getScore(A);
        }
        public int RemoveDuplicates(int[] nums)
        {
            // Dictionary<int,int> counter = new Dictionary<int, int>();
            int length = nums.Length;
            int number = Math.Min(2, length);
            for (int i = 0; i < length; i++)
            {
                int temp = nums[i];
                if (i < 2) continue;
                if (temp != nums[number - 2])
                {
                    nums[number] = temp;
                }
                //if (!counter.ContainsKey(temp))
                //{
                //    counter[temp] = 0;
                //}
                //if (counter[temp] < 2)
                //{
                //    counter[temp] += 1;
                //    nums[number] = temp;
                //    number += 1;
                //}
            }
            return number;
        }
        public int LeastBricks(List<List<int>> wall)
        {
            Dictionary<int, int> cache = new Dictionary<int, int>();
            int height = wall.Count;
            for (int i = 0; i < height; i++)
            {
                int sum = 0;
                for (int j = 0; j < wall[i].Count - 1; j++)
                {
                    sum += wall[i][j];
                    if (!cache.ContainsKey(sum)) cache[sum] = 0;
                    cache[sum] += 1;
                }
            }
            return cache.Values.Count > 0 ? height - cache.Values.Max() : height;
            //Dictionary<int, List<(int, int)>> cache = new Dictionary<int, List<(int, int)>>();
            //int height = wall.Count;
            //int progress = 0;
            //int maxprogress = 0;
            //cache[0] = new List<(int, int)>();
            //for (int r = 0; r < height; r++)
            //{
            //    cache[0].Add((r, 0));
            //}
            //int maxLength = 0;
            //while (progress <= maxprogress)
            //{
            //    if (!cache.ContainsKey(progress))
            //    {
            //        progress += 1;
            //        continue;
            //    }
            //    int length = cache[progress].Count;
            //    Console.WriteLine($"progress:{progress};length:{length};");
            //    if (progress != 0)
            //    {
            //        maxLength = Math.Max(maxLength, length);
            //    }
            //    for (int i = 0; i < length; i++)
            //    {
            //        (int row, int index) = cache[progress][i];

            //        if (index + 1 < wall[row].Count)
            //        {
            //            int brick = wall[row][index];
            //            if (!cache.ContainsKey(progress + brick))
            //            {
            //                cache[progress + brick] = new List<(int, int)>();
            //            }
            //            cache[progress + brick].Add((row, index + 1));
            //            maxprogress = Math.Max(maxprogress, progress + brick);
            //        }

            //    }
            //    cache.Remove(progress);
            //    progress += 1;
            //}
            //return height - maxLength;
        }
        public bool UniqueOccurrences(int[] arr)
        {
            Dictionary<int, int> counter = new Dictionary<int, int>();
            Dictionary<int, int> inverse = new Dictionary<int, int>();
            foreach (int num in arr)
            {
                if (!counter.ContainsKey(num)) counter[num] = 0;
                counter[num] += 1;
            }

            foreach (int count in counter.Values)
            {
                if (!inverse.ContainsKey(count)) inverse[count] = 0;
                inverse[count] += 1;
                if (inverse[count] > 1) return false;
            }

            return true;
        }
        public int[] GetRow(int[,] matrix, int rowNumber)
        {
            return Enumerable.Range(0, matrix.GetLength(1))
                    .Select(x => matrix[rowNumber, x])
                    .ToArray();
        }
        public int MinFallingPathSum(int[][] arr)
        {
            int height = arr.Length;
            int width = arr[0].Length;
            int[][] dp = new int[height][];
            for (int i = 0; i < height; i++)
            {
                dp[i] = new int[width];
            }
            for (int r = 0; r < height; r++)
            {
                List<int> row;
                if (r != 0) {
                    row = new List<int>(dp[r - 1]);
                }
                else
                {
                    row = new List<int>(dp[r]);
                }
                
                for (int c = 0; c < width; c++)
                {
                    if (r == 0)
                    {
                        dp[r][c] = arr[r][c];
                    } else
                    {
                        int temp = row[c];
                        row.RemoveAt(c);
                        dp[r][c] = arr[r][c] + row.Min();
                        row.Insert(c, temp);
                    }
                }
            }
            return new List<int>(dp[height-1]).Min();
        }
        public void BitOperation()
        {
            int c = 1;
            Console.WriteLine(Convert.ToString(c, toBase: 2));
            c = c | 2;
            Console.WriteLine(Convert.ToString(c, toBase: 2));
            c = c & 1;
            Console.WriteLine(Convert.ToString(c, toBase: 2));
            c = c << 2;
            Console.WriteLine(Convert.ToString(c, toBase: 2));
            c = c >> 2;
            Console.WriteLine(Convert.ToString(c, toBase: 2));
            c = c >> 2;
            Console.WriteLine(Convert.ToString(c, toBase: 2));
        }
        public void GameOfLife(int[][] board)
        {
            int[] dx = new int[8] { 0, 1, 1, 1, 0, -1, -1, -1 };
            int[] dy = new int[8] { -1, -1, 0, 1, 1, 1, 0, -1 };
            for (int y = 0; y < board.Length; y++)
            {
                for (int x = 0; x < board[0].Length; x++)
                {
                    int ally = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        int newx = x + dx[i];
                        int newy = y + dy[i];
                        if (newx >= 0 && newx < board[0].Length && newy >= 0 && newy < board.Length && (board[newy][newx] & 1) == 1) 
                        {
                            ally += 1;
                        }
                    }
                    if (ally == 2)
                    {
                        board[y][x] |= board[y][x] * 2;
                    }else if (ally == 3)
                    {
                        board[y][x] |= 2;
                    }
                }
            }
            for (int y = 0; y < board.Length; y++)
            {
                for (int x = 0; x < board[0].Length; x++)
                {
                    board[y][x] = board[y][x] >> 1;
                }
            }
        }
        public bool Search(int[] nums, int target)
        {
            int cur = 0;
            int last = nums[cur];
            if (target == last) return true;
            while (target > last)
            {
                cur += 1;
                if (cur > nums.Length - 1) return false;
                int now = nums[cur];
                if (now == target) return true;
                if (now < last) return false;
                last = now;
            }
            cur = nums.Length - 1;
            while (target < last)
            {
                if (cur < 0) return false;
                int now = nums[cur];
                if (now == target) return true;
                if (now > last) return false;
                last = now;
                cur -= 1;
            }
            return false;
        // 找到旋转点后二分
        }
        public string DecodeString(string s)
        {
            int start = s.IndexOf('[');
            int end = -1;
            if (start == -1) return s;
            int counter = 1;
            int digitstart = s.IndexOfAny(new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' });
            int repeat = int.Parse(s.Substring(digitstart, start - digitstart));
            for (int i = start + 1; i < s.Length; i++)
            {
                switch (s[i])
                {
                    case ']':
                        counter -= 1;
                        break;
                    case '[':
                        counter += 1;
                        break;
                    default:
                        break;
                }
                if (counter == 0)
                {
                    end = i;
                    break;
                }
            }
            string left = s.Substring(0, digitstart);
            string mid = DecodeString(s.Substring(start + 1, end - start - 1));
            for (int i = 0; i < repeat; i++)
            {
                left += mid;
            }
            string right = DecodeString(s.Substring(end + 1));
            return left + right;
        }
        public int FindMin(int[] nums)
        {
            int min = int.MinValue;
            foreach (var item in nums)
            {
                min = Math.Min(min, item);
            }
            return min;
        }
        public int BestSeqAtIndex(int[] height, int[] weight)
        {
            if (height.Length == 0) return 0;
            Array.Sort(height, weight);
            int count = 1;
            int last = weight[0];
            int cur = 1;
            while (cur < weight.Length)
            {
                if (weight[cur] > last)
                {
                    last = weight[cur];
                    count += 1;
                    cur += 1;
                }
                else
                {
                    if (count > 1)
                    {
                        if (cur + count - 1 < weight.Length && weight[cur + count - 1] < last)
                        {
                            
                        }
                        else
                        {
                            cur += 1;
                        }
                    } else
                    {
                        last = weight[cur];
                        cur += 1;
                    }
                }
                
            }
            return count;
        }
        bool IsBadVersion(int version)
        {
            return true;
        }
        public int FirstBadVersion(int n)
        {
            int low = 1;
            int high = n;
            while (low <= high)
            {
                int mid = low + ((high - low) >> 1);
                if (IsBadVersion(mid))
                {
                    high = mid - 1;
                } else
                {
                    low = mid + 1;
                }
            }
            return low;
        }
        public int SearchInsert(int[] nums, int target)
        {
            int low = 0;
            int high = nums.Length - 1;
            while (low <= high)
            {
                int mid = low + ((high - low) >> 1);
                if (nums[mid] == target) return mid;
                if (nums[mid] > target)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }
            return low;
        }
        public int[] SearchRange(int[] nums, int target)
        {
            // 没找到 [-1,-1]
            // 二分找两次，一次找下界，一次找上界
            // [5,7,7,8,8,10]
            // 先找下界
            int rangeLow = -1;
            int rangeHigh = -1;
            int low = 0;
            int high = nums.Length - 1;
            while (low <= high)
            {
                int mid = low + ((high - low) >> 1);
                if (nums[mid] >= target)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }
            if (low < nums.Length && nums[low] == target)
            {
                rangeLow = low;
            }
            // 找上界
            low = 0;
            high = nums.Length - 1;
            while (low <= high)
            {
                int mid = low + ((high - low) >> 1);
                Console.WriteLine($"{low} {mid} {high}");
                if (nums[mid] > target)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }
            if (high >= 0 && nums[high] == target)
            {
                rangeHigh = high;
            }

            return new int[2] { rangeLow, rangeHigh };
        }
        public double MyPow(double x, int n)
        {
            if (n == 1) return x;
            if (n == -1) return 1 / x;
            int left = n / 2;
            int right = n - left;
            return MyPow(x, left) * MyPow(x, right);
        }
        public bool BinarySearch2(int[] nums, int target)
        {
            int low = 0;
            int high = nums.Length - 1;
            while (low <= high)
            {
                int mid = low + ((high - low) >> 1);
                if (nums[mid] == target) return true;
                if (nums[mid] > target)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }
            return false;
        }
        public bool SearchMatrix2(int[][] matrix, int target)
        {
            int width = matrix[0].Length;
            for (int i = 0; i < matrix.Length; i++)
            {
                if (matrix[i][0] <= target && target <= matrix[i][width - 1])
                {
                    if (BinarySearch2(matrix[i], target)) return true;
                }
            }
            return false;
        }
        public bool SearchMatrix3(int[][] matrix, int target)
        {
            if (matrix.Length == 0) return false;
            if (matrix[0].Length == 0) return false;
            int col = 0;
            int row = matrix.Length - 1;
            while (row >= 0 && col <= matrix[0].Length - 1)
            {
                if (matrix[row][col] == target) return true;
                if (matrix[row][col] > target)
                {
                    row -= 1;
                }
                else
                {
                    col += 1;
                }
            }
            return false;
        }
        public static void Main()
        {
            Solution solution = new Solution();
            //var input = new List<List<int>>();
            //input.Add(new List<int> { 1, 1 });
            //input.Add(new List<int> { 2 });
            //input.Add(new List<int> { 1, 1 });

            //Console.WriteLine(output.ToString());
            // var output = solution.NumDecodings("111111111");
            // Console.WriteLine(output.ToString());
            // Console.WriteLine(output.ToString());
            solution.SearchRange(new int[] { 5, 7, 7, 8, 8, 10 }, 5);
            Console.ReadLine();
        }

    }
}
