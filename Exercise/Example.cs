/************************************************************************
*Copyright(c) 2021   All Rights Reserved .
*Author		：Feiyk
*CreatedOn  ：
*Email		：861168745@qq.com
*Describe   ：
*UseCase    ：
*Vesion     : 2021|V1.0.0.0 
***********************************************************************/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace Exercise
{
    class Example
    {
        // bfs
        public bool bfs(Complex begin, Complex end, HashSet<Complex> legalPositions)
        {
            var directions = new Complex[4] { new Complex(1, 0), new Complex(-1, 0), new Complex(0, 1), new Complex(0, -1) };
            HashSet<Complex> bfsSet = new HashSet<Complex>();
            bfsSet.Add(begin);
            while (bfsSet.Count > 0)
            {
                HashSet<Complex> tempSet = new HashSet<Complex>();
                foreach (Complex currentPosition in bfsSet)
                {
                    foreach (Complex direction in directions)
                    {
                        Complex nextPosition = currentPosition + direction;
                        if (legalPositions.Contains(nextPosition))
                        {
                            if (nextPosition == end) return true;
                            tempSet.Add(nextPosition);
                            legalPositions.Remove(nextPosition);
                        }
                    }
                }
                bfsSet = tempSet;
            }
            
            return false;
        }
        // 基础二分
        public int BinarySearch(int[] nums, int target)
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
                } else
                {
                    low = mid + 1;
                }
            }
            return -1;
        }
    }
}
