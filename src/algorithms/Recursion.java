package algorithms;

/**
 * Created by Biplob on 10/18/2017.
 */
public class Recursion {

    public static int fibR(int n) {
        if(n < 2) {
            return n;
        } else {
            return fibR(n-1) + fibR(n-2);
        }
    }

    public static int fibM(int n) {
        return fibMemo(n, new int[n+1]);
    }

    public static int fibMemo(int n, int[] memo) {
        if (n < 2) {
            return n;
        }

        if (memo[n] == 0) {
            memo[n] = fibMemo(n-1, memo) + fibMemo(n-2, memo);
        }

        return memo[n];
    }

    public static int fibMDp(int n) {
        if (n < 2) {
            return n;
        }

        int[] arr = new int[n+1];
        arr[0] = 0;
        arr[1] = 1;
        for (int i=2; i<=n; i++) {
            arr[i] = arr[i-1] + arr[i-2];
        }
        return arr[n];
    }

    public static int fibMI(int n) {
        if (n < 2) {
            return n;
        }

        int[] arr = {0,1};
        for (int i=2; i<=n; i++) {
            int val = arr[0] + arr[1];
            arr[0] = arr[1];
            arr[1] = val;
        }
        return arr[1];
    }
}
