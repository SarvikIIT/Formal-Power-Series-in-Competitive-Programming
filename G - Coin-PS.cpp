// HAR HAR MAHADEV
#pragma GCC optimize ("Ofast")
#pragma GCC target ("sse,sse2,mmx")
#pragma GCC optimize ("-ffloat-store")
#ifndef ONLINE_JUDGE
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

#define int long long
#define endl '\n'
#define all(x) (x).begin(), (x).end()
#define sz(x) ((int)(x).size())
#define pb push_back
#define ff first
#define ss second
#define rep(i,a,b) for(int i = a; i < b; i++)
#define repr(i,a,b) for(int i = a; i >= b; i--)
#define precise(i) cout<<fixed<<setprecision(i)
#define be(x) x.begin(), x.end()
#define each(a,x) for (auto& a : x)

using ll = long long;
using ld = long double;
using str = string;
using vi = vector<int>;
using vl = vector<ll>;
using vvi = vector<vi>;
using vvl = vector<vl>;
using pii = pair<int,int>;
using pll = pair<ll,ll>;
using vpii = vector<pii>;
using vpll = vector<pll>;

const ll MOD = 998244353;
const ll INF = 1e18;
const ll MOD1 = 998244353;
const int N = 1e5 + 5;

// ---------- modint ----------
template <const int32_t MOD_>
struct modint {
  int32_t value;

  modint() = default;

  // Normalize 32-bit input into [0, MOD)
  modint(int32_t v_) {
    int64_t v = static_cast<int64_t>(v_) % MOD_;
    if (v < 0) v += MOD_;
    value = static_cast<int32_t>(v);
  }

  // Normalize 64-bit input into [0, MOD)
  modint(int64_t v_) {
    int64_t v = v_ % MOD_;
    if (v < 0) v += MOD_;
    value = static_cast<int32_t>(v);
  }

  inline modint<MOD_> operator + (modint<MOD_> other) const {
    int32_t c = this->value + other.value;
    return modint<MOD_>(c >= MOD_ ? c - MOD_ : c);
  }
  inline modint<MOD_> operator - (modint<MOD_> other) const {
    int32_t c = this->value - other.value;
    return modint<MOD_>(c < 0 ? c + MOD_ : c);
  }
  inline modint<MOD_> operator * (modint<MOD_> other) const {
    int32_t c = (int64_t)this->value * other.value % MOD_;
    return modint<MOD_>(c < 0 ? c + MOD_ : c);
  }
  inline modint<MOD_> & operator += (modint<MOD_> other) {
    this->value += other.value;
    if (this->value >= MOD_) this->value -= MOD_;
    return *this;
  }
  inline modint<MOD_> & operator -= (modint<MOD_> other) {
    this->value -= other.value;
    if (this->value < 0) this->value += MOD_;
    return *this;
  }
  inline modint<MOD_> & operator *= (modint<MOD_> other) {
    this->value = (int64_t)this->value * other.value % MOD_;
    if (this->value < 0) this->value += MOD_;
    return *this;
  }
  inline modint<MOD_> operator - () const {
    return modint<MOD_>(this->value ? MOD_ - this->value : 0);
  }

  modint<MOD_> pow(uint64_t k) const {
    modint<MOD_> x = *this, y = 1;
    for (; k; k >>= 1) {
      if (k & 1) y *= x;
      x *= x;
    }
    return y;
  }
  // Optional: signed exponent convenience
  modint<MOD_> pow(long long k) const {
    return (k >= 0) ? pow((uint64_t)k) : inv().pow((uint64_t)(-k));
  }

  modint<MOD_> inv() const { return pow((long long)MOD_ - 2); }
  inline modint<MOD_> operator /  (modint<MOD_> other) const { return *this *  other.inv(); }
  inline modint<MOD_> operator /= (modint<MOD_> other)       { return *this *= other.inv(); }
  inline bool operator == (modint<MOD_> other) const { return value == other.value; }
  inline bool operator != (modint<MOD_> other) const { return value != other.value; }
  inline bool operator < (modint<MOD_> other) const { return value < other.value; }
  inline bool operator > (modint<MOD_> other) const { return value > other.value; }

  friend istream & operator >> (istream & in, modint<MOD_> &n) {
    long long v;
    in >> v;
    n = modint<MOD_>(v);
    return in;
  }
  friend ostream & operator << (ostream & out, modint<MOD_> n) {
    return out << n.value;
  }
};
template <int32_t MOD_> modint<MOD_> operator * (int64_t value, modint<MOD_> n) { return modint<MOD_>(value) * n; }
template <int32_t MOD_> modint<MOD_> operator * (int32_t value, modint<MOD_> n) { return modint<MOD_>(value % MOD_) * n; }

using mint = modint<MOD>;

inline void fastscan(int &number) {
    bool negative = false;
    int c;
    number = 0;
    c = getchar();
    if (c=='-') {
        negative = true;
        c = getchar();
    }
    for (; (c>47 && c<58); c=getchar())
        number = number *10 + c - 48;
    if (negative)
        number *= -1;
}
    template<class T>
    using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

    struct OrderedMultiSet {
        using P = pair<ll,int>;
        tree<P, null_type, less<P>, rb_tree_tag, tree_order_statistics_node_update> os;
        int uid = 0;
        void insert(ll x){ os.insert({x, uid++}); }
        void erase_one(ll x){ auto it = os.lower_bound({x, -1}); if(it != os.end() && it->ff == x) os.erase(it); }
        int order_of_key(ll x){ return os.order_of_key({x, -1}); }
        ll find_by_order_ll(int k){ return os.find_by_order(k)->ff; }
        int size() const { return (int)os.size(); }
    };
// I AM DSS LOVER :)

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
template<typename T1, typename T2>
using safe_map = unordered_map<T1, T2, custom_hash>;

template<typename T>
using safe_set = unordered_set<T, custom_hash>;

template<typename K>
using safe_unordered_set = unordered_set<K, custom_hash>;

template<typename K, typename V>
using safe_unordered_map = unordered_map<K, V, custom_hash>;

// Function to take input of an array
template<typename T>
inline void take(vector<T>& arr, int size) {
    arr.resize(size);
    rep(i,0,size) cin >> arr[i];
}

// Function to take input of a 2D array
template<typename T>
inline void take2D(vector<vector<T>>& arr, int start_row, int end_row, int start_col, int end_col) {
    rep(i, start_row, end_row) {
        rep(j, start_col, end_col) {
            cin >> arr[i][j];
        }
    }
}

inline ll mod(ll x) { return ((x % MOD + MOD) % MOD); }
inline ll power(ll x, ll y) {
    ll res = 1;
    x = x % MOD;
    while (y > 0) {
        if (y & 1) res = (res * x) % MOD;
        x = (x * x) % MOD;
        y = y >> 1;
    }
    return res;
}

inline ll gcd(ll a, ll b) {
    if (b > a) return gcd(b, a);
    if (b == 0) return a;
    return gcd(b, a % b);
}
inline ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }
inline bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

vector<ll> fact(N), invfact(N);
void precompute_factorials() {
    fact[0] = 1;
    for(int i = 1; i < N; i++) {
        fact[i] = (fact[i-1] * i) % MOD;
    }
    invfact[N-1] = power(fact[N-1], MOD-2);
    for(int i = N-2; i >= 0; i--) {
        invfact[i] = (invfact[i+1] * (i+1)) % MOD;
    }
}
inline ll nCr(int n, int r) {
    if(r > n || r < 0) return 0;
    return (fact[n] * ((invfact[r] * invfact[n-r]) % MOD)) % MOD;
}
// ---------- Geometry (2D) ----------
const long double EPS = 1e-12L;
int sgn(long double x){ return (x>EPS) - (x<-EPS); }
struct P {
    long double x, y;
    P() : x(0), y(0) {}
    P(long double x, long double y): x(x), y(y) {}
    P operator+(const P& o) const { return P(x+o.x, y+o.y); }
    P operator-(const P& o) const { return P(x-o.x, y-o.y); }
    P operator*(long double k) const { return P(x*k, y*k); }
    P operator/(long double k) const { return P(x/k, y/k); }
    bool operator<(const P& o) const { return x<o.x-EPS || (fabsl(x-o.x)<=EPS && y<o.y-EPS); }
    bool operator==(const P& o) const { return fabsl(x-o.x)<=EPS && fabsl(y-o.y)<=EPS; }
};
/*
        .. / .-.. --- ...- . / -.. .. -.-- .- / ... .. -. --. .... / ... .- -.-. .... -.. . ...- .-


*/
inline long double dot(const P& a, const P& b){ return a.x*b.x + a.y*b.y; }
inline long double cross(const P& a, const P& b){ return a.x*b.y - a.y*b.x; }
inline long double cross(const P& a, const P& b, const P& c){ return cross(b-a, c-a); }
inline long double norm2(const P& a){ return dot(a,a); }
inline long double norm(const P& a){ return sqrtl(norm2(a)); }
int orient(const P& a, const P& b, const P& c){ return sgn(cross(b-a, c-a)); }
bool onSeg(const P& a, const P& b, const P& p){ if (sgn(cross(b-a, p-a))!=0) return false; return sgn(dot(p-a, p-b))<=0; }
bool segInterProper(const P& a, const P& b, const P& c, const P& d){
    int o1 = orient(a,b,c), o2 = orient(a,b,d);
    int o3 = orient(c,d,a), o4 = orient(c,d,b);
    if (o1==0 && onSeg(a,b,c)) return true;
    if (o2==0 && onSeg(a,b,d)) return true;
    if (o3==0 && onSeg(c,d,a)) return true;
    if (o4==0 && onSeg(c,d,b)) return true;
    return (o1*o2<0 && o3*o4<0);
}
bool lineInter(const P& a, const P& b, const P& c, const P& d, P& out){
    P r = b-a, s = d-c;
    long double rxs = cross(r,s);
    if (fabsl(rxs) <= EPS) return false;
    long double t = cross(c-a, s) / rxs;
    out = a + r*t;
    return true;
}
vector<P> convexHull(vector<P> v){
    sort(all(v));
    v.erase(unique(all(v)), v.end());
    if (sz(v)<=1) return v;
    vector<P> lo, up;
    for (auto p: v){
        while (sz(lo)>=2 && orient(lo[sz(lo)-2], lo.back(), p) <= 0) lo.pop_back();
        lo.pb(p);
    }
    for (int i=(int)v.size()-1;i>=0;i--){
        auto p=v[i];
        while (sz(up)>=2 && orient(up[sz(up)-2], up.back(), p) <= 0) up.pop_back();
        up.pb(p);
    }
    lo.pop_back(); up.pop_back();
    lo.insert(lo.end(), all(up));
    return lo;
}
long double polygonArea(const vector<P>& poly){
    long double s = 0;
    for (int i=0;i<sz(poly);i++){
        int j=(i+1)%sz(poly);
        s += cross(poly[i], poly[j]);
    }
    return fabsl(s)/2.0L;
}


class SegmentTree {
private:
    vi ar, segment_tree, lazy;
    int n;

public:
    SegmentTree(vi& arr) {
        ar = arr;
        n = sz(arr);
        segment_tree.resize(4 * n, INF);
        lazy.resize(4 * n, 0);
        buildTree(0, n - 1, 1);
    }

    void buildTree(int start_index, int end_index, int segment_index) {
        if (start_index == end_index) {
            segment_tree[segment_index] = ar[start_index];
            return;
        }

        int mid = (start_index + end_index) / 2;
        buildTree(start_index, mid, 2 * segment_index);
        buildTree(mid + 1, end_index, 2 * segment_index + 1);

        segment_tree[segment_index] = min(segment_tree[2 * segment_index], segment_tree[2 * segment_index + 1]);
    }

    void push(int node, int start, int end) {
        if (lazy[node] != 0) {
            segment_tree[node] += lazy[node];
            if (start != end) {
                lazy[2 * node] += lazy[node];
                lazy[2 * node + 1] += lazy[node];
            }
            lazy[node] = 0;
        }
    }

    void rangeUpdate(int node, int start, int end, int l, int r, int val) {
        push(node, start, end);
        if (start > end || start > r || end < l) return;
        if (start >= l && end <= r) {
            lazy[node] += val;
            push(node, start, end);
            return;
        }
        int mid = (start + end) / 2;
        rangeUpdate(2 * node, start, mid, l, r, val);
        rangeUpdate(2 * node + 1, mid + 1, end, l, r, val);
        segment_tree[node] = min(segment_tree[2 * node], segment_tree[2 * node + 1]);
    }

    int query(int start_index, int end_index, int segment_index, int query_start, int query_end) {
        push(segment_index, start_index, end_index);
        if (query_end < start_index || query_start > end_index)
            return INF;
        if (query_start <= start_index && end_index <= query_end)
            return segment_tree[segment_index];

        int mid = (start_index + end_index) / 2;
        int left = query(start_index, mid, 2 * segment_index, query_start, query_end);
        int right = query(mid + 1, end_index, 2 * segment_index + 1, query_start, query_end);

        return min(left, right);
    }

    int rangeMin(int l, int r) {
        return query(0, n - 1, 1, l, r);
    }

    void updateRange(int l, int r, int val) {
        rangeUpdate(1, 0, n - 1, l, r, val);
    }
};
struct LCA {
    vector<int> height, euler, first;
    vector<bool> visited;
    int n, lg;
    vector<vector<int>> st;

    LCA(vector<vector<int>> &adj, int root = 0) {
        n = adj.size();
        height.resize(n);
        first.resize(n);
        euler.reserve(n * 2);
        visited.assign(n, false);
        dfs(adj, root);
        sparseTable();
    }

    void dfs(vector<vector<int>> &adj, int node, int h = 0) {
        visited[node] = true;
        height[node] = h;
        first[node] = euler.size();
        euler.push_back(node);
        for (auto to : adj[node]) {
            if (!visited[to]) {
                dfs(adj, to, h + 1);
                euler.push_back(node);
            }
        }
    }

    void sparseTable() {
        int m = euler.size();
        lg = __lg(m);
        st.assign(lg+1, vector<int>(m, INT_MAX));
        for (int j = 0; j < m; j++) st[0][j] = j;
        for (int i = 1; i <= lg; i++)
            for (int j = 0; j + (1<<i) <= m; j++) {
                int a = st[i-1][j];
                int b = st[i-1][j + (1<<(i-1))];
                st[i][j] = (height[euler[a]] < height[euler[b]]) ? a : b;
            }
    }

    int query(int u, int v) {
        int l = first[u], r = first[v];
        if (l > r)
            swap(l, r);
        int k = __lg(r-l+1);
        int a = st[k][l];
        int b = st[k][r - (1ll<<k) + 1];
        return (height[euler[a]] < height[euler[b]]) ? euler[a] : euler[b];
    }
};

class DSU {
private:
    vi parent, rank, size;

public:
    DSU(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        size.resize(n, 1);
        rep(i, 0, n) parent[i] = i;
    }

    inline int find(int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }

    inline void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;

        if (rank[px] < rank[py]) {
            parent[px] = py;
            size[py] += size[px];
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
            size[px] += size[py];
        } else {
            parent[py] = px;
            rank[px]++;
            size[px] += size[py];
        }
    }

    inline bool same(int x, int y) {
        return find(x) == find(y);
    }

    inline int getSize(int x) {
        return size[find(x)];
    }
};

inline bool isPalindrome(string s) {
    int n = sz(s);
    rep(i, 0, n/2) if(s[i] != s[n-1-i]) return false;
    return true;
}

inline string toBinary(int n) {
    string binary = "";
    while (n > 0) {
        binary = char('0' + n % 2) + binary;
        n /= 2;
    }
    return binary.empty() ? "0" : binary;
}

inline int fromBinary(string binary) {
    int num = 0;
    for (char c : binary) {
        num = num * 2 + (c - '0');
    }
    return num;
}

inline bool isValid(int i, int j, int n, int m) {
    return i >= 0 && i < n && j >= 0 && j < m;
}

const vi dx = {-1, 0, 1, 0, -1, -1, 1, 1};
const vi dy = {0, 1, 0, -1, -1, 1, -1, 1};

inline int countBits(int n) {
    return __builtin_popcountll(n);
}

inline int lowestBit(int n) {
    return n & (-n);
}

inline bool isPowerOfTwo(int n) {
    return n && !(n & (n - 1));
}

template<typename T>
inline void sortVector(vector<T>& arr) {
    sort(all(arr));
}

template<typename T>
inline void reverseVector(vector<T>& arr) {
    reverse(all(arr));
}


#ifndef DEBUG_TEMPLATE_CPP
#define DEBUG_TEMPLATE_CPP
// #define cerr cout
namespace __DEBUG_UTIL__
{
    using namespace std;

    template <typename T>
    struct is_iterable_impl {
        template <typename U>
        static auto test(int) -> decltype(begin(declval<U>()), end(declval<U>()), true_type{});
        template <typename U>
        static false_type test(...);
        using type = decltype(test<T>(0));
    };

    template <typename T>
    using is_iterable = typename is_iterable_impl<T>::type;

    void print(const char *x) { cerr << x; }
    void print(char x) { cerr << "\'" << x << "\'"; }
    void print(bool x) { cerr << (x ? "T" : "F"); }
    void print(string x) { cerr << "\"" << x << "\""; }
    void print(vector<bool> &v)
    { /* Overloaded this because stl optimizes vector<bool> by using
         _Bit_reference instead of bool to conserve space. */
        int f = 0;
        cerr << '{';
        for (auto &&i : v)
            cerr << (f++ ? "," : "") << (i ? "T" : "F");
        cerr << "}";
    }
    template <typename T>
    void print(T &&x)
    {
        if constexpr (is_iterable<T>::value)
            if (size(x) && is_iterable<decltype(*(begin(x)))>::value)
            { /* Iterable inside Iterable */
                int f = 0;
                cerr << "\n~~~~~\n";
                for (auto &&i : x)
                {
                    cerr << setw(2) << left << f++, print(i), cerr << "\n";
                }
                cerr << "~~~~~\n";
            }
            else
            { /* Normal Iterable */
                int f = 0;
                cerr << "{";
                for (auto &&i : x)
                    cerr << (f++ ? "," : ""), print(i);
                cerr << "}";
            }
        else
            cerr << x;
    }
    template <typename T, typename... V>
    void printer(const char *names, T &&head, V &&...tail)
    {
        int i = 0;
        for (size_t bracket = 0; names[i] != '\0' and (names[i] != ',' or bracket != 0); i++)
            if (names[i] == '(' or names[i] == '<' or names[i] == '{')
                bracket++;
            else if (names[i] == ')' or names[i] == '>' or names[i] == '}')
                bracket--;
        cerr.write(names, i) << " = ";
        print(head);
        if constexpr (sizeof...(tail))
            cerr << " ||", printer(names + i + 1, tail...);
        else
            cerr << "]\n";
    }
    template <typename T, typename... V>
    void printerArr(const char *names, T arr[], size_t N, V... tail)
    {
        size_t i = 0;
        for (; names[i] and names[i] != ','; i++)
            cerr << names[i];
        for (i++; names[i] and names[i] != ','; i++)
            ;
        cerr << " = {";
        for (size_t ind = 0; ind < N; ind++)
            cerr << (ind ? "," : ""), print(arr[ind]);
        cerr << "}";
        if constexpr (sizeof...(tail))
            cerr << " ||", printerArr(names + i + 1, tail...);
        else
            cerr << "]\n";
    }

}
#ifndef ONLINE_JUDGE
#define debug(...) std::cerr << __LINE__ << ": [", __DEBUG_UTIL__::printer(#__VA_ARGS__, __VA_ARGS__)
#define debugArr(...) std::cerr << __LINE__ << ": [", __DEBUG_UTIL__::printerArr(#__VA_ARGS__, __VA_ARGS__)
#else
#define debug(...)
#define debugArr(...)
#endif
#endif

int n,m,l;

void solve() {
    cin>>n>>m>>l;
    vector<mint> f(n+1);
    f[0] = 1;
    /*
        Recurrence boils down to:
        Given we can choose coins as 1 + x^a ... hence it becomes 1/(1-x^a) and since we can choose a from m to m+l-1 hence:
        Ans(m+1) = Ans(m) * (1-x^m)/(1-x^(m+l))
        now we just need to maintain f array for current m and then we can get f array for m+1 by using this recurrence and printing out f[m]
    */
    auto add = [&](int a) -> void {
        rep(i,0,n-a+1){
            f[i+a] += f[i];
        }
    };
    auto rm = [&](int a) -> void {
        repr(i,n-a,0){
            f[i+a] -= f[i];
        }
    };
    rep(i,1,l){
        add(i);
    }
    rep(i,l,m+1){
        add(i);
        cout<<f[n]<<endl;
        rm(i-l+1);
    }
}

signed main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    auto start = chrono::high_resolution_clock::now();


    // precompute_factorials();

    int t = 1;
    // cin >> t;
    while(t--) solve();


    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cerr << "Execution time: " << duration.count() << " μs" << endl;



    return 0;
}
