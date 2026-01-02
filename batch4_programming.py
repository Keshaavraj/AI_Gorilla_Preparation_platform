"""
Batch 4: Programming Questions
- OOP (15 questions)
- Algorithms (20 questions)
- REST APIs (12 questions)
"""

from database_manager import QuestionDatabase, create_question

def populate_oop():
    """15 OOP Questions"""
    questions = [
        create_question(
            "What is the purpose of encapsulation in object-oriented programming?",
            [
                "To make code run faster",
                "To bundle data and methods that operate on that data, hiding internal details",
                "To create multiple classes",
                "To enable inheritance"
            ],
            1,
            "Encapsulation bundles data (attributes) and methods together in a class while hiding internal implementation details. This is achieved through access modifiers (private, protected, public) and provides controlled access via getters/setters. Benefits: (1) data protection, (2) reduced coupling, (3) easier maintenance. Example: making attributes private and providing public methods to access them.",
            "Medium",
            85
        ),
        create_question(
            "In Python, what is the difference between class variables and instance variables?",
            [
                "They are identical",
                "Class variables are shared by all instances; instance variables are unique to each instance",
                "Class variables are faster",
                "Instance variables can't be modified"
            ],
            1,
            "Class variables are defined in the class body and shared by all instances: changes affect all. Instance variables are defined in __init__ with self.var and unique to each instance. Access: ClassName.class_var for class variables, instance.var for instance variables. Class variables are useful for constants or counters shared across instances. Instance variables store object-specific state.",
            "Medium",
            90
        ),
        create_question(
            "What is polymorphism in OOP?",
            [
                "Creating multiple classes",
                "The ability of objects of different classes to respond to the same method call in different ways",
                "Making classes private",
                "Inheriting from multiple parents"
            ],
            1,
            "Polymorphism allows different classes to implement the same interface differently. Two types: (1) Compile-time (method overloading), (2) Runtime (method overriding via inheritance). Example: Shape classes (Circle, Square) each implement draw() differently. Enables writing generic code that works with any Shape. Duck typing in Python: 'if it walks like a duck and quacks like a duck, it's a duck' - focus on methods, not types.",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of the __init__ method in Python classes?",
            [
                "To delete the object",
                "Constructor method that initializes object state when an instance is created",
                "To print the object",
                "To make the class abstract"
            ],
            1,
            "__init__ is the constructor called automatically when creating an instance. It initializes instance variables. Syntax: def __init__(self, params): self.attribute = value. self refers to the instance being created. __init__ returns None (shouldn't return anything). For cleanup, use __del__ (destructor). __new__ is the actual object creator (rarely overridden).",
            "Medium",
            75
        ),
        create_question(
            "What is the difference between inheritance and composition?",
            [
                "They are the same",
                "Inheritance is 'is-a' relationship (subclass extends parent); composition is 'has-a' relationship (object contains other objects)",
                "Inheritance is always better",
                "Composition is deprecated"
            ],
            1,
            "Inheritance: class Dog(Animal) - Dog IS-A Animal, inherits Animal's methods. Composition: class Car contains Engine - Car HAS-AN Engine. Composition is often preferred (favor composition over inheritance) as it's more flexible, reduces tight coupling, and avoids deep inheritance hierarchies. Use inheritance for true 'is-a' relationships, composition for 'has-a' or 'uses-a'. Multiple composition vs. multiple inheritance issues.",
            "Hard",
            95
        ),
        create_question(
            "What is method overriding in OOP?",
            [
                "Creating multiple methods with same name in one class",
                "Subclass providing specific implementation of method already defined in parent class",
                "Deleting parent methods",
                "Making methods private"
            ],
            1,
            "Method overriding allows a subclass to provide specific implementation of a method inherited from parent. The overridden method in subclass has same name and signature. When called on subclass instance, subclass version executes. Use super().method() to call parent version. Enables polymorphism - different behavior based on actual object type at runtime. Different from overloading (same name, different parameters).",
            "Medium",
            85
        ),
        create_question(
            "What is an abstract class?",
            [
                "A class that is difficult to understand",
                "A class that cannot be instantiated and serves as a base template for subclasses",
                "A class with no methods",
                "A class that is deprecated"
            ],
            1,
            "Abstract classes define interfaces that subclasses must implement. Cannot be instantiated directly. In Python, use ABC module: class MyClass(ABC): @abstractmethod def my_method(): pass. Subclasses must implement all abstract methods. Use abstract classes to enforce a contract - ensuring all subclasses have required methods. Different from interfaces (pure abstract classes with no implementation).",
            "Hard",
            95
        ),
        create_question(
            "What does the @property decorator do in Python?",
            [
                "Makes a method into a property with getter/setter behavior",
                "Deletes the method",
                "Makes the method static",
                "Speeds up execution"
            ],
            0,
            "@property converts a method into a getter, allowing attribute-like access. @property def x(self): return self._x allows obj.x instead of obj.x(). Provide setter with @x.setter def x(self, value): self._x = value. This enables encapsulation - control access to attributes while maintaining clean syntax. Can add validation in setters. Use _variable convention for internal attributes.",
            "Medium",
            90
        ),
        create_question(
            "What is the purpose of __str__ and __repr__ methods in Python?",
            [
                "They delete objects",
                "__str__ for human-readable string (print); __repr__ for unambiguous representation (debugging)",
                "They are identical",
                "__str__ is deprecated"
            ],
            1,
            "__str__ should return a user-friendly string (used by str() and print()). __repr__ should return an unambiguous string ideally usable to recreate the object (used by repr() and in interactive shell). Best practice: __repr__ for developers, __str__ for end users. If only one, define __repr__ (used as fallback for __str__). Example: __repr__ = 'Point(1, 2)', __str__ = '(1, 2)'.",
            "Medium",
            85
        ),
        create_question(
            "What is multiple inheritance and what problem does it create?",
            [
                "Inheriting from one class",
                "Inheriting from multiple parent classes; creates diamond problem when parents share a common ancestor",
                "Creating multiple objects",
                "A deprecated feature"
            ],
            1,
            "Multiple inheritance: class C(A, B) inherits from both A and B. Diamond problem: if A and B inherit from Base, which Base version does C use? Python solves this with MRO (Method Resolution Order) using C3 linearization. Check with Class.__mro__ or Class.mro(). super() follows MRO. While powerful, multiple inheritance can be complex - prefer composition or mixins. Use for mixins (adding functionality).",
            "Hard",
            100
        ),
        create_question(
            "What is the difference between @staticmethod and @classmethod in Python?",
            [
                "They are identical",
                "staticmethod doesn't receive implicit first argument; classmethod receives class as first argument (cls)",
                "staticmethod is deprecated",
                "classmethod is faster"
            ],
            1,
            "@staticmethod: no implicit first argument, can't access instance or class. Use for utility functions related to the class. @classmethod: receives class as first argument (cls), can access/modify class state. Use for factory methods. Instance methods receive instance (self). Examples: @staticmethod def utility(x): ...; @classmethod def from_string(cls, s): return cls(...) creates instance from string.",
            "Hard",
            95
        ),
        create_question(
            "What is the Liskov Substitution Principle (LSP)?",
            [
                "Lists should be substituted with arrays",
                "Objects of a subclass should be replaceable with objects of the superclass without breaking the application",
                "A sorting algorithm",
                "A Python built-in function"
            ],
            1,
            "LSP states: if S is a subtype of T, objects of type T can be replaced with objects of type S without altering program correctness. Subclasses must honor the contract of parent class. Violations: subclass throwing new exceptions, strengthening preconditions, weakening postconditions. Ensures inheritance is used correctly. Example: if Square inherits Rectangle but can't set width independently, it violates LSP.",
            "Hard",
            95
        ),
        create_question(
            "What is the Single Responsibility Principle (SRP)?",
            [
                "A class should have only one method",
                "A class should have only one reason to change - one responsibility",
                "A class should inherit from one parent only",
                "A class should have one instance"
            ],
            1,
            "SRP (from SOLID): a class should have one responsibility - one reason to change. This improves maintainability and reduces coupling. Bad: UserClass handling authentication, database, and email. Good: separate UserAuth, UserRepository, EmailService classes. Each class does one thing well. Makes code easier to test, understand, and modify. Applies to functions too - single, well-defined purpose.",
            "Medium",
            90
        ),
        create_question(
            "What is dependency injection?",
            [
                "Deleting dependencies",
                "Providing dependencies to a class from outside rather than creating them internally",
                "A type of inheritance",
                "A Python library"
            ],
            1,
            "Dependency Injection: pass dependencies to a class rather than creating them inside. Instead of class A: def __init__(self): self.b = B(), use class A: def __init__(self, b): self.b = b. Benefits: (1) loose coupling, (2) easier testing (inject mocks), (3) flexibility (swap implementations). DI container frameworks automate this. Related to Dependency Inversion Principle (depend on abstractions, not concretions).",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of interfaces in OOP (or abstract base classes in Python)?",
            [
                "To slow down code",
                "To define a contract that implementing classes must follow, ensuring consistent APIs",
                "To create GUI interfaces",
                "To connect to the internet"
            ],
            1,
            "Interfaces define method signatures that implementing classes must provide - a contract. In Python, use ABC and @abstractmethod. Benefits: (1) enforces consistent API, (2) enables polymorphism, (3) documents expected behavior. Example: PaymentProcessor interface ensures all payment implementations have process_payment(). Use when multiple classes should share the same interface but have different implementations.",
            "Medium",
            85
        )
    ]
    return questions


def populate_algorithms():
    """20 Algorithms Questions"""
    questions = [
        create_question(
            "What is the time complexity of binary search?",
            [
                "O(n)",
                "O(log n)",
                "O(n log n)",
                "O(1)"
            ],
            1,
            "Binary search has O(log n) time complexity. It divides the search space in half each iteration, working only on sorted arrays. After k iterations, search space is n/2^k. When n/2^k = 1, k = log₂(n). Space: O(1) iterative, O(log n) recursive (call stack). Much faster than linear search O(n) for large datasets. Prerequisite: sorted array.",
            "Medium",
            75
        ),
        create_question(
            "What is the difference between BFS (Breadth-First Search) and DFS (Depth-First Search)?",
            [
                "They are identical",
                "BFS explores level by level using a queue; DFS explores as deep as possible using a stack",
                "BFS is always faster",
                "DFS can't find paths"
            ],
            1,
            "BFS uses queue (FIFO): explores all neighbors before going deeper, finds shortest path in unweighted graphs. DFS uses stack (LIFO) or recursion: explores as far as possible before backtracking. BFS better for shortest path, DFS for existence checks/topological sort. Space: BFS O(width), DFS O(height). Time: both O(V+E) for graphs. Use BFS for shortest path, DFS for cycle detection, topological sort.",
            "Hard",
            95
        ),
        create_question(
            "What is the time complexity of quicksort in the average case?",
            [
                "O(n)",
                "O(n log n)",
                "O(n²)",
                "O(log n)"
            ],
            1,
            "Quicksort average case: O(n log n). Worst case: O(n²) when pivot is always smallest/largest (e.g., already sorted). Best case: O(n log n). Space: O(log n) for recursion stack. Randomized pivot selection avoids worst case in practice. In-place sorting (unlike merge sort). Despite worst case, often faster than merge sort due to better cache performance and in-place operation.",
            "Medium",
            85
        ),
        create_question(
            "What data structure should you use to implement a LRU (Least Recently Used) cache efficiently?",
            [
                "Array only",
                "Hash map + doubly linked list",
                "Binary tree",
                "Stack"
            ],
            1,
            "LRU cache requires O(1) get and O(1) put. Solution: Hash map for O(1) access + doubly linked list for O(1) reordering. Hash map stores key→node, linked list maintains access order (most recent at head). On access: move node to head. On capacity: remove tail. Hash map alone can't track order efficiently, linked list alone can't find keys quickly. OrderedDict in Python implements this pattern.",
            "Hard",
            100
        ),
        create_question(
            "What is dynamic programming?",
            [
                "Programming that changes at runtime",
                "Solving problems by breaking them into overlapping subproblems and storing results to avoid recomputation",
                "A programming language",
                "Parallel programming"
            ],
            1,
            "Dynamic Programming (DP) solves optimization problems by: (1) breaking into overlapping subproblems, (2) storing results (memoization or tabulation) to avoid recomputation. Key: optimal substructure + overlapping subproblems. Approaches: top-down (memoization/recursion), bottom-up (tabulation/iteration). Classic examples: Fibonacci, knapsack, longest common subsequence. Transforms exponential problems to polynomial. Different from divide-and-conquer (non-overlapping subproblems).",
            "Hard",
            95
        ),
        create_question(
            "What is the time complexity of inserting an element at the beginning of a linked list vs. an array?",
            [
                "Both O(1)",
                "Linked list O(1), array O(n)",
                "Linked list O(n), array O(1)",
                "Both O(n)"
            ],
            1,
            "Linked list: O(1) - create new node, point to current head, update head. Array: O(n) - shift all elements right to make space at index 0. This is why linked lists excel at insertions/deletions at ends, while arrays provide O(1) random access. Dynamic arrays (Python list) amortize append to O(1) but prepend remains O(n). For frequent prepends, use deque (doubly-linked list with O(1) operations at both ends).",
            "Medium",
            85
        ),
        create_question(
            "What is a hash collision and how can it be resolved?",
            [
                "When two files have the same hash",
                "When two keys map to the same hash table index; resolved via chaining or open addressing",
                "A type of car accident",
                "An error in the algorithm"
            ],
            1,
            "Hash collision occurs when hash function maps different keys to the same index. Resolution methods: (1) Chaining - each bucket contains a linked list of colliding entries, (2) Open addressing - probe for next available slot (linear, quadratic, double hashing). Chaining allows more elements than slots, open addressing requires good probing. Python dict uses open addressing. Good hash function minimizes collisions.",
            "Hard",
            95
        ),
        create_question(
            "What is the time complexity of heapsort?",
            [
                "O(n)",
                "O(n log n)",
                "O(n²)",
                "O(log n)"
            ],
            1,
            "Heapsort: O(n log n) in all cases (best, average, worst). Build heap: O(n). Extract max n times, each O(log n) for heapify: n * log n. Space: O(1) as it's in-place. Advantages: guaranteed O(n log n), in-place. Disadvantages: not stable, worse cache performance than quicksort. Heap is also used for priority queue. Python heapq provides min-heap.",
            "Medium",
            80
        ),
        create_question(
            "What is memoization?",
            [
                "Remembering things manually",
                "Caching function results based on arguments to avoid recomputation",
                "A memory management technique",
                "Deleting old data"
            ],
            1,
            "Memoization caches function results keyed by arguments. On subsequent calls with same arguments, return cached result. Enables top-down DP. Example: Fibonacci with dict to store computed values. Python: use @lru_cache decorator. Trade-off: memory for speed. Only for pure functions (same inputs → same output). Different from tabulation (bottom-up DP). functools.lru_cache limits cache size.",
            "Medium",
            85
        ),
        create_question(
            "What is the difference between a min-heap and a max-heap?",
            [
                "They are identical",
                "Min-heap: parent ≤ children (root is minimum); max-heap: parent ≥ children (root is maximum)",
                "Min-heap is smaller in size",
                "Max-heap uses more memory"
            ],
            1,
            "Min-heap: parent node value ≤ children, root is minimum. Max-heap: parent ≥ children, root is maximum. Both complete binary trees (all levels full except possibly last, filled left-to-right). Operations: insert O(log n), extract min/max O(log n), peek O(1). Python heapq is min-heap (negate values for max-heap). Use min-heap for smallest element priority, max-heap for largest.",
            "Medium",
            80
        ),
        create_question(
            "What is a trie (prefix tree) used for?",
            [
                "Sorting numbers",
                "Efficient storage and retrieval of strings, especially for prefix searches",
                "Binary search",
                "Hashing"
            ],
            1,
            "Trie stores strings efficiently by sharing common prefixes. Each node represents a character, paths form words. Operations: insert/search/delete O(m) where m = string length. Space: O(ALPHABET_SIZE * m * n) for n strings. Use cases: autocomplete, spell checker, IP routing. Advantages over hash table: prefix queries, sorted order. Disadvantage: space overhead. Compressed variant: radix tree.",
            "Hard",
            95
        ),
        create_question(
            "What is the time complexity of merging two sorted arrays of size m and n?",
            [
                "O(m + n)",
                "O(m * n)",
                "O(log(m + n))",
                "O(max(m, n))"
            ],
            0,
            "Merging two sorted arrays: O(m + n) time, O(m + n) space for result. Algorithm: use two pointers, compare elements, take smaller, advance pointer. Must traverse all elements of both arrays exactly once. Cannot be faster than O(m+n) as you must examine all elements. This is the merge step in merge sort. In-place merging (given extra space in one array) is more complex but achievable.",
            "Medium",
            80
        ),
        create_question(
            "What is a balanced binary search tree and why is it important?",
            [
                "A tree with equal left and right subtrees",
                "A BST where height is O(log n), ensuring operations remain efficient",
                "A tree with all leaves at same level",
                "A sorting algorithm"
            ],
            1,
            "Balanced BST maintains height O(log n) for n nodes, ensuring search/insert/delete remain O(log n). Unbalanced BST can degrade to O(n) in worst case (e.g., inserting sorted data). Self-balancing implementations: AVL (strict balance), Red-Black (relaxed balance, faster updates), B-trees (for disk). Python: no built-in balanced BST (use bisect + list or sortedcontainers library). Crucial for databases and file systems.",
            "Hard",
            95
        ),
        create_question(
            "What is the space complexity of merge sort?",
            [
                "O(1)",
                "O(n)",
                "O(log n)",
                "O(n log n)"
            ],
            1,
            "Merge sort space: O(n) for auxiliary array used in merging. Time: O(n log n) all cases. Stable sort (preserves relative order of equal elements). Not in-place (unlike quicksort). Recursion depth: O(log n) call stack. Total space: O(n) + O(log n) = O(n). Good for linked lists (O(1) space), external sorting (large datasets on disk). Parallelizable.",
            "Medium",
            80
        ),
        create_question(
            "What is topological sort and when is it used?",
            [
                "Sorting numbers",
                "Linear ordering of directed acyclic graph (DAG) vertices respecting edge directions",
                "Sorting alphabetically",
                "A hash function"
            ],
            1,
            "Topological sort orders DAG vertices so for every edge u→v, u comes before v. Use cases: task scheduling with dependencies, build systems, course prerequisites. Algorithms: (1) DFS with stack, (2) Kahn's (BFS with in-degree). Only possible for DAGs (cyclic graphs have no topological order). Multiple valid orderings possible. Time: O(V+E). Detects cycles if unable to order all vertices.",
            "Hard",
            95
        ),
        create_question(
            "What is the difference between comparison-based and non-comparison-based sorting?",
            [
                "No difference",
                "Comparison-based uses comparisons (e.g., quicksort), lower bound O(n log n); non-comparison uses counting/radix, can be O(n)",
                "Comparison-based is always faster",
                "Non-comparison doesn't work"
            ],
            1,
            "Comparison-based (quicksort, mergesort, heapsort): O(n log n) lower bound proven. Non-comparison (counting sort, radix sort, bucket sort): can achieve O(n) by exploiting properties of input (e.g., limited range, digit-by-digit). Counting sort: O(n+k) for range k. Radix: O(d*n) for d digits. Bucket: O(n) average for uniformly distributed data. Trade-off: non-comparison needs assumptions about input.",
            "Hard",
            100
        ),
        create_question(
            "What is a greedy algorithm?",
            [
                "An algorithm that uses a lot of memory",
                "An algorithm that makes locally optimal choices at each step, hoping to find global optimum",
                "An algorithm that always fails",
                "An algorithm that runs slowly"
            ],
            1,
            "Greedy algorithms make locally optimal choice at each step, hoping for global optimum. Doesn't always work (needs greedy-choice property and optimal substructure). Examples that work: Dijkstra's shortest path, Huffman coding, Kruskal's MST. Counter-example: greedy fails for making change with arbitrary coin denominations. Faster than DP but less general. Prove correctness: show local choices lead to global optimum.",
            "Medium",
            90
        ),
        create_question(
            "What is the master theorem used for?",
            [
                "Solving linear equations",
                "Analyzing time complexity of divide-and-conquer recursive algorithms",
                "Finding shortest paths",
                "Sorting algorithms"
            ],
            1,
            "Master theorem analyzes recurrences of form T(n) = aT(n/b) + f(n) where a≥1, b>1. Compares f(n) to n^(log_b(a)). Three cases determine if dominated by: (1) leaves, (2) all levels equally, (3) root. Applies to binary search, merge sort, many divide-and-conquer algorithms. Doesn't handle all recurrences (e.g., T(n) = T(n-1) + n). Powerful tool for algorithmic analysis.",
            "Hard",
            95
        ),
        create_question(
            "What is the Floyd-Warshall algorithm used for?",
            [
                "Sorting arrays",
                "Finding shortest paths between all pairs of vertices in a weighted graph",
                "Binary search",
                "String matching"
            ],
            1,
            "Floyd-Warshall finds shortest paths between all vertex pairs in weighted graph (positive or negative edges, but no negative cycles). Time: O(V³), Space: O(V²). DP algorithm: for each intermediate vertex k, check if path through k is shorter. Simpler than running Dijkstra V times. Detects negative cycles. Returns distance matrix. Use when you need all-pairs shortest paths.",
            "Hard",
            95
        ),
        create_question(
            "What is the difference between divide-and-conquer and dynamic programming?",
            [
                "They are identical",
                "D&C has non-overlapping subproblems; DP has overlapping subproblems (stores results to avoid recomputation)",
                "D&C is always slower",
                "DP doesn't use recursion"
            ],
            1,
            "Both break problems into subproblems. Divide-and-conquer: subproblems are independent (e.g., merge sort splits array, subproblems don't overlap). DP: subproblems overlap (e.g., Fibonacci: fib(n-1) and fib(n-2) both compute fib(n-3)), so cache results. DP requires optimal substructure + overlapping subproblems. D&C solves each subproblem once. DP avoids recomputation through memoization/tabulation.",
            "Hard",
            100
        )
    ]
    return questions


def populate_rest_apis():
    """12 REST APIs Questions"""
    questions = [
        create_question(
            "What does REST stand for and what is its key principle?",
            [
                "Random Execution State Transfer",
                "Representational State Transfer - stateless client-server communication",
                "Rapid Execution System Transfer",
                "Remote Execution Service Transfer"
            ],
            1,
            "REST (Representational State Transfer) is an architectural style for distributed systems. Key principles: (1) Stateless - each request contains all necessary information, (2) Client-Server separation, (3) Cacheable responses, (4) Uniform interface, (5) Layered system. REST APIs use HTTP methods (GET, POST, PUT, DELETE) on resources identified by URIs. Stateless means no client context stored on server between requests.",
            "Medium",
            85
        ),
        create_question(
            "What is the difference between PUT and PATCH HTTP methods?",
            [
                "They are identical",
                "PUT replaces entire resource; PATCH partially updates resource",
                "PUT is for creation only",
                "PATCH is deprecated"
            ],
            1,
            "PUT is idempotent and replaces the entire resource - send complete representation. PATCH is for partial updates - send only changed fields. Example: PUT /users/1 with {name, email, age} replaces all fields. PATCH /users/1 with {email} updates only email. PUT idempotent: multiple identical requests have same effect. PATCH may not be idempotent depending on implementation. Use PUT for full updates, PATCH for partial.",
            "Hard",
            90
        ),
        create_question(
            "What does a 401 Unauthorized HTTP status code indicate?",
            [
                "Server error",
                "Authentication required or failed - client must authenticate",
                "Resource not found",
                "Request succeeded"
            ],
            1,
            "401 Unauthorized means authentication is required or has failed. Client must provide valid credentials. Actually should be called 'Unauthenticated'. Different from 403 Forbidden (authenticated but lacks permission). Response should include WWW-Authenticate header indicating authentication method. Common in APIs requiring API keys, OAuth tokens, or basic auth. Fix: provide valid credentials in Authorization header.",
            "Medium",
            80
        ),
        create_question(
            "What is the purpose of HTTP status code 204 No Content?",
            [
                "Error occurred",
                "Request succeeded but no content to return (common for DELETE)",
                "Content not found",
                "Partial content"
            ],
            1,
            "204 No Content indicates successful request with no response body. Common for: (1) DELETE operations (successfully deleted, nothing to return), (2) PUT/PATCH where response body isn't needed. Status 200 would include response body. Saves bandwidth when response body isn't necessary. Different from 404 (not found) or 201 (created with resource in body). Client shouldn't expect body with 204.",
            "Medium",
            75
        ),
        create_question(
            "What is idempotency in REST APIs?",
            [
                "Making requests faster",
                "Making the same request multiple times produces the same result",
                "Encrypting requests",
                "Compressing responses"
            ],
            1,
            "Idempotent operations produce the same result regardless of how many times they're executed. Idempotent HTTP methods: GET, PUT, DELETE, HEAD, OPTIONS. Non-idempotent: POST. Example: DELETE /user/1 multiple times - first deletes, rest do nothing (resource stays deleted). PUT /user/1 {data} multiple times - resource stays in same state. Important for retries and reliability. POST creates new resource each time (not idempotent).",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of the Accept header in HTTP requests?",
            [
                "To accept cookies",
                "To specify what media types (content types) the client can handle in response",
                "To authorize the request",
                "To compress data"
            ],
            1,
            "Accept header specifies media types client can process in response. Example: Accept: application/json requests JSON format. Accept: application/xml for XML. Accept: */* accepts any format. Server uses content negotiation to return appropriate format or 406 Not Acceptable if it can't provide requested format. Related: Content-Type (what you're sending), Accept-Language (preferred language).",
            "Medium",
            80
        ),
        create_question(
            "What is CORS (Cross-Origin Resource Sharing)?",
            [
                "A database technology",
                "A mechanism that allows restricted resources to be requested from another domain",
                "A programming language",
                "A sorting algorithm"
            ],
            1,
            "CORS allows controlled access to resources from different origins (domain, protocol, or port). Browser security policy normally blocks cross-origin requests. Server includes CORS headers (Access-Control-Allow-Origin, etc.) to permit. Preflight requests (OPTIONS) check permissions before actual request. Common issue: API doesn't include CORS headers, browser blocks request. Configured server-side. Essential for web apps calling external APIs.",
            "Hard",
            90
        ),
        create_question(
            "What is the difference between authentication and authorization?",
            [
                "They are the same",
                "Authentication verifies identity (who you are); authorization verifies permissions (what you can do)",
                "Authorization comes first",
                "Authentication is deprecated"
            ],
            1,
            "Authentication (AuthN): verifying identity - 'Who are you?' - login with username/password, API key, OAuth. Authorization (AuthZ): verifying permissions - 'What can you do?' - checking if authenticated user can access resource. Flow: authenticate first, then authorize. HTTP: 401 for authentication failure, 403 for authorization failure. Example: login (authenticate), then check if user can delete post (authorize).",
            "Medium",
            85
        ),
        create_question(
            "What is rate limiting in APIs and why is it important?",
            [
                "Making APIs slower",
                "Restricting number of requests a client can make in a time period to prevent abuse",
                "Speeding up responses",
                "Compressing data"
            ],
            1,
            "Rate limiting restricts requests per time window (e.g., 100/minute, 1000/hour) to: (1) prevent abuse/DoS, (2) ensure fair usage, (3) control costs. Common headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. Status 429 Too Many Requests when exceeded. Algorithms: token bucket, leaky bucket, fixed/sliding window. Important for public APIs. Implement per-user or per-IP.",
            "Medium",
            85
        ),
        create_question(
            "What is the purpose of API versioning?",
            [
                "To make APIs slower",
                "To manage changes and maintain backward compatibility while evolving API",
                "To delete old endpoints",
                "To compress responses"
            ],
            1,
            "API versioning allows introducing breaking changes without disrupting existing clients. Strategies: (1) URI versioning (/v1/users, /v2/users), (2) Header versioning (Accept: application/vnd.api+json;version=1), (3) Query parameter (?version=1). URI versioning most common. Allows: deprecating old versions gradually, supporting multiple versions simultaneously. Important for public APIs with many clients. Minimize versions - maintain only necessary ones.",
            "Medium",
            80
        ),
        create_question(
            "What is the difference between HTTP methods GET and POST?",
            [
                "They are identical",
                "GET retrieves data, is idempotent, cached; POST creates/submits data, not idempotent, not cached",
                "GET is deprecated",
                "POST is for deletion"
            ],
            1,
            "GET: retrieves resources, idempotent, cacheable, params in URL (query string), should not modify server state. POST: creates resources or submits data, not idempotent, not cacheable, params in body, modifies server state. GET limited by URL length, POST not limited. GET bookmarkable/linkable. Use GET for reads, POST for writes/creates. Security: don't send sensitive data in GET (URLs logged). GET requests shouldn't have side effects.",
            "Medium",
            75
        ),
        create_question(
            "What is a RESTful resource and how should it be named?",
            [
                "A random endpoint",
                "An entity or concept in the system, named with nouns (not verbs) in plural form",
                "A function name",
                "A database table"
            ],
            1,
            "Resources are entities/concepts (users, products, orders). Naming: (1) use nouns, not verbs (GET /users, not /getUsers), (2) plural form (/users), (3) hierarchical (/users/123/orders), (4) lowercase with hyphens (/order-items). HTTP methods provide the verbs (GET, POST, PUT, DELETE). Good: GET /users/123, POST /users. Bad: GET /getUser?id=123, POST /createUser. Resources should represent domain concepts clearly.",
            "Medium",
            80
        )
    ]
    return questions


if __name__ == "__main__":
    db = QuestionDatabase()

    print("Populating OOP questions...")
    db.add_bulk_questions("OOP", populate_oop())
    print(f"✓ Added {len(populate_oop())} OOP questions")

    print("Populating Algorithms questions...")
    db.add_bulk_questions("Algorithms", populate_algorithms())
    print(f"✓ Added {len(populate_algorithms())} Algorithms questions")

    print("Populating REST APIs questions...")
    db.add_bulk_questions("REST APIs", populate_rest_apis())
    print(f"✓ Added {len(populate_rest_apis())} REST APIs questions")

    stats = db.get_statistics()
    print(f"\n{'='*60}")
    print(f"BATCH 4 COMPLETE - Programming")
    print(f"{'='*60}")
    print(f"Total questions in database: {stats['total_questions']}")
    print("\nBatch 4 questions by category:")
    for category in ["OOP", "Algorithms", "REST APIs"]:
        count = db.get_question_count(category)
        print(f"  {category}: {count} questions")
    print(f"\nDatabase saved to: questions_db.json")
