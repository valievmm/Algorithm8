"""
Лабораторная работа №8
Система управления рецептами — Recipe Management System
=========================================================
Структуры данных:
  • HashTable     — поиск рецепта по ID за O(1)
  • AVLTree       — хранение рецептов, отсортированных по рейтингу / названию
  • Graph         — граф похожих рецептов (рекомендации)

Алгоритмы:
  • BFS           — рекомендации по похожим рецептам
  • QuickSort     — сортировка отчётов
  • Dijkstra      — поиск «ближайшего» рецепта по весам рёбер графа

Веб-интерфейс: Flask (один файл, templates встроены как строки)
"""

from __future__ import annotations
import time
import random
import string
from collections import deque
from typing import Any, Generator, Optional
import heapq

# ─────────────────────────────────────────────
# 1. СТРУКТУРЫ ДАННЫХ
# ─────────────────────────────────────────────

# ---------- 1.1 Hash Table (separate chaining) ----------

class HashTable:
    """Хеш-таблица с цепочками; средний O(1) вставка/поиск/удаление."""

    def __init__(self, size: int = 128):
        self._size = size
        self._buckets: list[list] = [[] for _ in range(size)]
        self._count = 0

    def _hash(self, key: str) -> int:
        h = 5381
        for ch in str(key):
            h = (h * 33 + ord(ch)) & 0xFFFFFFFF
        return h % self._size

    def insert(self, key: str, value: Any) -> None:
        idx = self._hash(key)
        for i, (k, _) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx][i] = (key, value)
                return
        self._buckets[idx].append((key, value))
        self._count += 1
        if self._count / self._size > 0.75:
            self._rehash()

    def find(self, key: str) -> Optional[Any]:
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return None

    def delete(self, key: str) -> bool:
        idx = self._hash(key)
        for i, (k, _) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx].pop(i)
                self._count -= 1
                return True
        return False

    def values(self) -> list:
        result = []
        for bucket in self._buckets:
            for _, v in bucket:
                result.append(v)
        return result

    def items(self) -> list:
        result = []
        for bucket in self._buckets:
            for k, v in bucket:
                result.append((k, v))
        return result

    def _rehash(self) -> None:
        old_buckets = self._buckets
        self._size *= 2
        self._buckets = [[] for _ in range(self._size)]
        self._count = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.insert(k, v)

    def __len__(self) -> int:
        return self._count


# ---------- 1.2 AVL Tree ----------

class _AVLNode:
    __slots__ = ("key", "payload", "left", "right", "height")

    def __init__(self, key, payload=None):
        self.key = key
        self.payload = payload          # список артикулов / id с этим ключом
        self.left: Optional[_AVLNode] = None
        self.right: Optional[_AVLNode] = None
        self.height: int = 1


class AVLTree:
    """
    AVL-дерево с ключом (comparable).
    payload — список значений при одинаковых ключах (мультисет).
    """

    def __init__(self):
        self._root: Optional[_AVLNode] = None

    # --- helpers ---
    @staticmethod
    def _h(node: Optional[_AVLNode]) -> int:
        return node.height if node else 0

    @staticmethod
    def _bf(node: _AVLNode) -> int:
        return AVLTree._h(node.left) - AVLTree._h(node.right)

    @staticmethod
    def _upd(node: _AVLNode) -> None:
        node.height = 1 + max(AVLTree._h(node.left), AVLTree._h(node.right))

    # --- rotations ---
    @staticmethod
    def _rot_right(y: _AVLNode) -> _AVLNode:
        x = y.left
        y.left = x.right
        x.right = y
        AVLTree._upd(y)
        AVLTree._upd(x)
        return x

    @staticmethod
    def _rot_left(x: _AVLNode) -> _AVLNode:
        y = x.right
        x.right = y.left
        y.left = x
        AVLTree._upd(x)
        AVLTree._upd(y)
        return y

    @staticmethod
    def _balance(node: _AVLNode) -> _AVLNode:
        AVLTree._upd(node)
        bf = AVLTree._bf(node)
        if bf > 1:
            if AVLTree._bf(node.left) < 0:
                node.left = AVLTree._rot_left(node.left)
            return AVLTree._rot_right(node)
        if bf < -1:
            if AVLTree._bf(node.right) > 0:
                node.right = AVLTree._rot_right(node.right)
            return AVLTree._rot_left(node)
        return node

    # --- public API ---
    def insert(self, key, value_id: str) -> None:
        self._root = self._insert(self._root, key, value_id)

    def _insert(self, node: Optional[_AVLNode], key, vid: str) -> _AVLNode:
        if not node:
            n = _AVLNode(key)
            n.payload = [vid]
            return n
        if key == node.key:
            if vid not in node.payload:
                node.payload.append(vid)
            return node
        elif key < node.key:
            node.left = self._insert(node.left, key, vid)
        else:
            node.right = self._insert(node.right, key, vid)
        return self._balance(node)

    def delete(self, key, value_id: str) -> None:
        self._root = self._delete(self._root, key, value_id)

    def _delete(self, node: Optional[_AVLNode], key, vid: str) -> Optional[_AVLNode]:
        if not node:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key, vid)
        elif key > node.key:
            node.right = self._delete(node.right, key, vid)
        else:
            node.payload = [p for p in node.payload if p != vid]
            if not node.payload:
                # удаляем узел
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                # заменяем минимальным правого поддерева
                mn = self._min_node(node.right)
                node.key = mn.key
                node.payload = mn.payload
                node.right = self._delete_min(node.right)
        return self._balance(node) if node else None

    @staticmethod
    def _min_node(node: _AVLNode) -> _AVLNode:
        while node.left:
            node = node.left
        return node

    @staticmethod
    def _delete_min(node: _AVLNode) -> Optional[_AVLNode]:
        if not node.left:
            return node.right
        node.left = AVLTree._delete_min(node.left)
        return AVLTree._balance(node)

    def inorder(self) -> Generator:
        """Возвращает (key, value_id) в порядке возрастания ключа."""
        yield from self._inorder(self._root)

    def _inorder(self, node: Optional[_AVLNode]) -> Generator:
        if not node:
            return
        yield from self._inorder(node.left)
        for vid in node.payload:
            yield node.key, vid
        yield from self._inorder(node.right)


# ---------- 1.3 Граф (список смежности) ----------

class Graph:
    """Взвешенный неориентированный граф."""

    def __init__(self):
        self._adj: dict[str, dict[str, float]] = {}

    def add_node(self, node: str) -> None:
        if node not in self._adj:
            self._adj[node] = {}

    def remove_node(self, node: str) -> None:
        if node in self._adj:
            for nb in list(self._adj[node]):
                self._adj[nb].pop(node, None)
            del self._adj[node]

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = weight
        self._adj[v][u] = weight

    def remove_edge(self, u: str, v: str) -> None:
        self._adj.get(u, {}).pop(v, None)
        self._adj.get(v, {}).pop(u, None)

    def has_node(self, node: str) -> bool:
        return node in self._adj

    def neighbors(self, node: str) -> dict[str, float]:
        return self._adj.get(node, {})

    def all_edges(self) -> list[tuple[str, str, float]]:
        seen = set()
        result = []
        for u, nbs in self._adj.items():
            for v, w in nbs.items():
                key = tuple(sorted((u, v)))
                if key not in seen:
                    seen.add(key)
                    result.append((u, v, w))
        return result

    def nodes(self) -> list[str]:
        return list(self._adj.keys())


# ─────────────────────────────────────────────
# 2. АЛГОРИТМЫ
# ─────────────────────────────────────────────

def quicksort(arr: list, key=None, reverse: bool = False) -> list:
    """Быстрая сортировка (in-place на копии)."""
    arr = list(arr)
    _qs(arr, 0, len(arr) - 1, key or (lambda x: x))
    if reverse:
        arr.reverse()
    return arr


def _qs(arr, lo, hi, key):
    if lo < hi:
        p = _partition(arr, lo, hi, key)
        _qs(arr, lo, p - 1, key)
        _qs(arr, p + 1, hi, key)


def _partition(arr, lo, hi, key):
    pivot = key(arr[hi])
    i = lo - 1
    for j in range(lo, hi):
        if key(arr[j]) <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1


def bfs(graph: Graph, start: str, max_depth: int = 2) -> list[str]:
    """BFS до глубины max_depth, возвращает узлы (без стартового)."""
    if not graph.has_node(start):
        return []
    visited = {start}
    queue = deque([(start, 0)])
    result = []
    while queue:
        node, depth = queue.popleft()
        if 0 < depth <= max_depth:
            result.append(node)
        if depth >= max_depth:
            continue
        for nb in graph.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return result


def dijkstra(graph: Graph, start: str) -> dict[str, float]:
    """Кратчайшие расстояния от start до всех узлов (веса рёбер)."""
    dist = {n: float("inf") for n in graph.nodes()}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.neighbors(u).items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


# ─────────────────────────────────────────────
# 3. МОДЕЛЬ ДАННЫХ
# ─────────────────────────────────────────────

class Recipe:
    """Рецепт."""

    def __init__(self, recipe_id: str, name: str, category: str,
                 ingredients: list[str], steps: list[str],
                 time_min: int, rating: float = 5.0):
        self.id = recipe_id
        self.name = name
        self.category = category
        self.ingredients = ingredients      # список строк
        self.steps = steps                  # список строк
        self.time_min = time_min
        self.rating = round(max(0.0, min(10.0, float(rating))), 1)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "ingredients": self.ingredients,
            "steps": self.steps,
            "time_min": self.time_min,
            "rating": self.rating,
        }

    def __repr__(self) -> str:
        return f"Recipe({self.id}, {self.name!r}, {self.rating})"


# ─────────────────────────────────────────────
# 4. DATA MANAGER — ядро системы
# ─────────────────────────────────────────────

class DataManager:
    """
    Интегрирует HashTable, два AVL-дерева (по рейтингу и по названию) и граф.
    """

    def __init__(self):
        self.hash_table = HashTable(size=256)
        self.tree_by_rating = AVLTree()     # ключ: rating (float)
        self.tree_by_name = AVLTree()       # ключ: name (str)
        self.graph = Graph()                # граф похожих рецептов

    # --- CRUD ---

    def add_recipe(self, recipe: Recipe) -> None:
        if self.hash_table.find(recipe.id):
            raise ValueError(f"Рецепт с ID '{recipe.id}' уже существует.")
        self.hash_table.insert(recipe.id, recipe)
        self.tree_by_rating.insert(recipe.rating, recipe.id)
        self.tree_by_name.insert(recipe.name.lower(), recipe.id)
        self.graph.add_node(recipe.id)

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        return self.hash_table.find(recipe_id)

    def update_recipe(self, recipe_id: str, **kwargs) -> Recipe:
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            raise KeyError(f"Рецепт '{recipe_id}' не найден.")
        if "rating" in kwargs and kwargs["rating"] != recipe.rating:
            self.tree_by_rating.delete(recipe.rating, recipe_id)
            recipe.rating = round(float(kwargs.pop("rating")), 1)
            self.tree_by_rating.insert(recipe.rating, recipe_id)
        if "name" in kwargs and kwargs["name"] != recipe.name:
            self.tree_by_name.delete(recipe.name.lower(), recipe_id)
            recipe.name = kwargs.pop("name")
            self.tree_by_name.insert(recipe.name.lower(), recipe_id)
        for k, v in kwargs.items():
            if hasattr(recipe, k):
                setattr(recipe, k, v)
        return recipe

    def delete_recipe(self, recipe_id: str) -> None:
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            raise KeyError(f"Рецепт '{recipe_id}' не найден.")
        self.hash_table.delete(recipe_id)
        self.tree_by_rating.delete(recipe.rating, recipe_id)
        self.tree_by_name.delete(recipe.name.lower(), recipe_id)
        self.graph.remove_node(recipe_id)

    # --- Сортированные списки (обход AVL) ---

    def get_sorted_by_rating(self, reverse: bool = True) -> list[Recipe]:
        pairs = list(self.tree_by_rating.inorder())
        recipes = [self.hash_table.find(rid) for _, rid in pairs if self.hash_table.find(rid)]
        if reverse:
            recipes.reverse()
        return recipes

    def get_sorted_by_name(self) -> list[Recipe]:
        pairs = list(self.tree_by_name.inorder())
        return [self.hash_table.find(rid) for _, rid in pairs if self.hash_table.find(rid)]

    # --- Граф: связи ---

    def add_relation(self, id1: str, id2: str, weight: float = 1.0) -> None:
        if not self.graph.has_node(id1):
            raise KeyError(f"Рецепт '{id1}' не найден.")
        if not self.graph.has_node(id2):
            raise KeyError(f"Рецепт '{id2}' не найден.")
        self.graph.add_edge(id1, id2, weight)

    def remove_relation(self, id1: str, id2: str) -> None:
        self.graph.remove_edge(id1, id2)

    # --- Рекомендации (BFS) ---

    def recommend(self, recipe_id: str, max_depth: int = 2) -> list[Recipe]:
        ids = bfs(self.graph, recipe_id, max_depth)
        return [r for rid in ids if (r := self.hash_table.find(rid))]

    # --- Dijkstra: ближайший по графу ---

    def closest(self, recipe_id: str) -> list[tuple[Recipe, float]]:
        dist = dijkstra(self.graph, recipe_id)
        result = []
        for rid, d in dist.items():
            if rid != recipe_id and d < float("inf"):
                r = self.hash_table.find(rid)
                if r:
                    result.append((r, d))
        result.sort(key=lambda x: x[1])
        return result

    # --- Отчёт (QuickSort) ---

    def generate_report(self, sort_by: str = "rating", reverse: bool = True) -> list[Recipe]:
        all_recipes = [r for r in self.hash_table.values() if isinstance(r, Recipe)]
        key_map = {
            "rating": lambda r: r.rating,
            "name": lambda r: r.name.lower(),
            "time": lambda r: r.time_min,
        }
        key_func = key_map.get(sort_by, key_map["rating"])
        return quicksort(all_recipes, key=key_func, reverse=reverse)

    # --- Поиск (простой) ---

    def search(self, query: str) -> list[Recipe]:
        q = query.lower()
        return [r for r in self.hash_table.values()
                if isinstance(r, Recipe) and (
                    q in r.name.lower() or
                    q in r.category.lower() or
                    any(q in ing.lower() for ing in r.ingredients)
                )]

    # --- Поиск по нескольким ингредиентам (новый метод) ---

    def search_by_ingredients(self, ingredients: list[str], mode: str = "AND") -> list[Recipe]:
        """Поиск рецептов по списку ингредиентов (подстрока). mode='AND' — все ингредиенты, 'OR' — любой."""
        all_recipes = [r for r in self.hash_table.values() if isinstance(r, Recipe)]
        result = []
        for r in all_recipes:
            if mode.upper() == "AND":
                if all(any(q in ing.lower() for ing in r.ingredients) for q in ingredients):
                    result.append(r)
            else:  # OR
                if any(any(q in ing.lower() for ing in r.ingredients) for q in ingredients):
                    result.append(r)
        return result

    # --- Нагрузочное тестирование ---

    def benchmark(self, n: int = 1000) -> dict:
        """Добавляет N рецептов и замеряет производительность."""
        import random, string, time

        def rand_id():
            return "bench_" + "".join(random.choices(string.ascii_lowercase, k=8))

        ids = []
        t0 = time.perf_counter()
        for _ in range(n):
            rid = rand_id()
            r = Recipe(
                rid,
                name="Recipe " + rid,
                category=random.choice(["Суп", "Салат", "Десерт", "Основное"]),
                ingredients=["ингр1", "ингр2"],
                steps=["шаг1"],
                time_min=random.randint(5, 120),
                rating=round(random.uniform(1, 10), 1),
            )
            self.add_recipe(r)
            ids.append(rid)
        insert_time = (time.perf_counter() - t0) * 1_000_000 / n  # мкс на вставку

        t0 = time.perf_counter()
        for rid in random.choices(ids, k=min(1000, n)):
            self.hash_table.find(rid)
        search_time = (time.perf_counter() - t0) * 1_000_000 / min(1000, n)

        t0 = time.perf_counter()
        self.get_sorted_by_rating()
        sort_tree_time = (time.perf_counter() - t0) * 1_000_000

        t0 = time.perf_counter()
        self.generate_report()
        quicksort_time = (time.perf_counter() - t0) * 1_000_000

        # Добавим несколько рёбер для BFS
        for i in range(min(10, n - 1)):
            try:
                self.add_relation(ids[i], ids[i + 1], weight=1.0)
            except Exception:
                pass
        t0 = time.perf_counter()
        if ids:
            bfs(self.graph, ids[0], max_depth=2)
        bfs_time = (time.perf_counter() - t0) * 1_000_000

        # Очистка тестовых данных
        for rid in ids:
            try:
                self.delete_recipe(rid)
            except Exception:
                pass

        return {
            "n": n,
            "insert_us": round(insert_time, 2),
            "search_us": round(search_time, 2),
            "sort_tree_us": round(sort_tree_time, 2),
            "quicksort_us": round(quicksort_time, 2),
            "bfs_us": round(bfs_time, 2),
        }


# ─────────────────────────────────────────────
# 5. ПРЕДЗАПОЛНЕНИЕ ДЕМО-ДАННЫМИ
# ─────────────────────────────────────────────

def _seed(dm: DataManager) -> None:
    recipes = [
        Recipe("r001", "Борщ классический", "Суп",
               ["свёкла", "капуста", "картофель", "морковь", "томатная паста", "мясо"],
               ["Сварить бульон.", "Обжарить свёклу с морковью.", "Добавить все овощи и варить 30 мин."],
               90, 9.2),
        Recipe("r002", "Цезарь с курицей", "Салат",
               ["куриное филе", "листья романо", "пармезан", "гренки", "соус Цезарь"],
               ["Обжарить курицу.", "Нарвать листья.", "Смешать с соусом и гренками."],
               25, 8.7),
        Recipe("r003", "Тирамису", "Десерт",
               ["маскарпоне", "яйца", "сахар", "эспрессо", "савоярди", "какао"],
               ["Взбить желтки с сахаром.", "Добавить маскарпоне.", "Намочить савоярди и собрать слоями."],
               40, 9.5),
        Recipe("r004", "Паста карбонара", "Основное",
               ["спагетти", "гуанчале", "яйца", "пекорино романо", "чёрный перец"],
               ["Сварить пасту.", "Обжарить гуанчале.", "Смешать яйца с сыром, добавить к пасте."],
               30, 9.1),
        Recipe("r005", "Греческий салат", "Салат",
               ["огурцы", "помидоры", "оливки", "фета", "красный лук", "оливковое масло"],
               ["Нарезать овощи.", "Добавить фету и оливки.", "Заправить маслом."],
               15, 8.3),
        Recipe("r006", "Том Ям", "Суп",
               ["креветки", "кокосовое молоко", "лемонграсс", "лайм", "грибы шиитаке", "чили"],
               ["Приготовить бульон с лемонграссом.", "Добавить кокосовое молоко.", "Закинуть креветки и грибы."],
               45, 9.4),
        Recipe("r007", "Шоколадный фондан", "Десерт",
               ["тёмный шоколад", "сливочное масло", "яйца", "сахар", "мука"],
               ["Растопить шоколад с маслом.", "Взбить яйца с сахаром.", "Смешать и выпекать 12 минут."],
               30, 9.6),
        Recipe("r008", "Пицца Маргарита", "Основное",
               ["тесто для пиццы", "томатный соус", "моцарелла", "базилик"],
               ["Раскатать тесто.", "Нанести соус и сыр.", "Выпекать при 250°C 10 минут."],
               40, 9.0),
    ]

    for r in recipes:
        dm.add_recipe(r)

    # Связи (похожие рецепты)
    dm.add_relation("r001", "r006", 1.0)   # Борщ ↔ Том Ям (оба супы)
    dm.add_relation("r002", "r005", 1.5)   # Цезарь ↔ Греческий
    dm.add_relation("r003", "r007", 1.0)   # Тирамису ↔ Фондан (десерты)
    dm.add_relation("r004", "r008", 1.5)   # Карбонара ↔ Пицца (итальянские)
    dm.add_relation("r004", "r002", 2.0)
    dm.add_relation("r007", "r003", 1.0)
    dm.add_relation("r006", "r001", 1.0)


# ─────────────────────────────────────────────
# 6. FLASK ВЕБ-ПРИЛОЖЕНИЕ
# ─────────────────────────────────────────────

try:
    from flask import Flask, request, jsonify, render_template_string
except ImportError:
    raise SystemExit("Flask не установлен. Запустите: pip install flask")

app = Flask(__name__)
dm = DataManager()
_seed(dm)

# ── HTML шаблон ──────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🍽 Система управления рецептами</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--surface:#1a1d27;--card:#22273a;
  --accent:#f97316;--accent2:#6366f1;
  --text:#e2e8f0;--muted:#94a3b8;--border:#2d3349;
  --green:#22c55e;--red:#ef4444;--yellow:#eab308;
}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
a{color:var(--accent);text-decoration:none}
/* Layout */
.layout{display:flex;min-height:100vh}
.sidebar{width:220px;background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;padding:1rem 0;flex-shrink:0}
.sidebar .logo{padding:.5rem 1.2rem 1.2rem;font-size:1.2rem;font-weight:700;color:var(--accent)}
.sidebar .logo span{color:var(--text)}
.nav-item{padding:.7rem 1.2rem;cursor:pointer;color:var(--muted);display:flex;align-items:center;gap:.6rem;
  transition:background .15s,color .15s;border-left:3px solid transparent}
.nav-item:hover,.nav-item.active{background:var(--card);color:var(--text);border-left-color:var(--accent)}
.main{flex:1;padding:1.5rem 2rem;overflow-y:auto}
/* Cards */
.page{display:none}.page.active{display:block}
.page-title{font-size:1.5rem;font-weight:700;margin-bottom:1.2rem;display:flex;align-items:center;gap:.5rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.2rem;margin-bottom:1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem}
.recipe-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem;
  transition:border-color .2s,transform .2s;cursor:pointer}
.recipe-card:hover{border-color:var(--accent);transform:translateY(-2px)}
.recipe-card .rc-name{font-weight:700;font-size:1rem;margin-bottom:.4rem}
.recipe-card .rc-cat{font-size:.78rem;color:var(--muted);margin-bottom:.5rem}
.recipe-card .rc-meta{display:flex;gap.8rem;align-items:center;margin-top:.7rem}
.badge{display:inline-flex;align-items:center;gap:.25rem;font-size:.75rem;
  padding:.18rem .55rem;border-radius:20px;font-weight:600}
.badge-green{background:rgba(34,197,94,.15);color:var(--green)}
.badge-orange{background:rgba(249,115,22,.15);color:var(--accent)}
.badge-purple{background:rgba(99,102,241,.15);color:var(--accent2)}
/* Stars */
.stars{color:var(--yellow);font-size:.85rem}
/* Form */
.form-group{margin-bottom:1rem}
label{display:block;font-size:.85rem;color:var(--muted);margin-bottom:.35rem}
input,textarea,select{width:100%;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:.6rem .8rem;color:var(--text);font-size:.9rem;outline:none;
  transition:border-color .2s}
input:focus,textarea:focus,select:focus{border-color:var(--accent)}
textarea{min-height:80px;resize:vertical}
.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.55rem 1.1rem;
  border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer;border:none;
  transition:opacity .2s,transform .1s}
.btn:hover{opacity:.88;transform:translateY(-1px)}
.btn:active{transform:translateY(0)}
.btn-primary{background:var(--accent);color:#fff}
.btn-secondary{background:var(--surface);color:var(--text);border:1px solid var(--border)}
.btn-danger{background:rgba(239,68,68,.2);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.btn-sm{padding:.35rem .75rem;font-size:.8rem}
/* Table */
table{width:100%;border-collapse:collapse}
th,td{padding:.6rem .8rem;text-align:left;border-bottom:1px solid var(--border);font-size:.875rem}
th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase}
tr:hover td{background:rgba(255,255,255,.02)}
/* Alert */
.alert{padding:.75rem 1rem;border-radius:8px;margin-bottom:1rem;font-size:.9rem}
.alert-ok{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:var(--green)}
.alert-err{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:var(--red)}
/* Modal */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:100;
  align-items:center;justify-content:center}
.modal-bg.open{display:flex}
.modal{background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:1.5rem;width:min(90vw,540px);max-height:85vh;overflow-y:auto}
.modal h3{font-size:1.1rem;margin-bottom:1rem}
/* Search */
.search-bar{display:flex;gap:.6rem;margin-bottom:1.2rem}
.search-bar input{flex:1}
/* Stats */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.8rem;margin-bottom:1.2rem}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem}
.stat-card .val{font-size:1.6rem;font-weight:700;color:var(--accent)}
.stat-card .lbl{font-size:.78rem;color:var(--muted);margin-top:.2rem}
/* Bench */
.bench-table td:first-child{color:var(--muted)}
/* Graph */
#graph-canvas{width:100%;height:400px;background:var(--surface);border:1px solid var(--border);
  border-radius:12px;display:block}
/* Scrollbar */
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
/* Util */
.row{display:flex;gap:.7rem;flex-wrap:wrap}
.mt{margin-top:1rem}.mb{margin-bottom:1rem}
.text-muted{color:var(--muted);font-size:.85rem}
.flex-between{display:flex;justify-content:space-between;align-items:center}
.chip{display:inline-block;background:var(--surface);border:1px solid var(--border);
  border-radius:20px;padding:.15rem .6rem;font-size:.75rem;margin:.15rem}
</style>
</head>
<body>
<div class="layout">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="logo">🍽 <span>Рецепты</span></div>
    <div class="nav-item active" data-page="home">🏠 Главная</div>
    <div class="nav-item" data-page="recipes">📋 Все рецепты</div>
    <div class="nav-item" data-page="add">➕ Добавить</div>
    <div class="nav-item" data-page="search">🔍 Поиск</div>
    <div class="nav-item" data-page="report">📊 Отчёт</div>
    <div class="nav-item" data-page="relations">🔗 Связи</div>
    <div class="nav-item" data-page="recommend">💡 Рекомендации</div>
    <div class="nav-item" data-page="dijkstra">📡 Dijkstra</div>
    <div class="nav-item" data-page="bench">⚡ Тестирование</div>
  </aside>

  <!-- Main -->
  <main class="main">

    <!-- HOME -->
    <div class="page active" id="page-home">
      <div class="page-title">🏠 Панель управления</div>
      <div class="stats-grid" id="home-stats"></div>
      <div class="card">
        <div style="font-weight:700;margin-bottom:.8rem">Топ рецептов по рейтингу</div>
        <div id="home-top"></div>
      </div>
    </div>

    <!-- RECIPES -->
    <div class="page" id="page-recipes">
      <div class="flex-between mb">
        <div class="page-title" style="margin:0">📋 Все рецепты</div>
        <div class="row">
          <select id="sort-by" style="width:auto">
            <option value="rating">По рейтингу ↓</option>
            <option value="name">По названию ↑</option>
            <option value="time">По времени ↑</option>
          </select>
          <button class="btn btn-secondary btn-sm" onclick="loadRecipes()">Обновить</button>
        </div>
      </div>
      <div class="grid" id="recipes-grid"></div>
    </div>

    <!-- ADD -->
    <div class="page" id="page-add">
      <div class="page-title">➕ Добавить рецепт</div>
      <div class="card" style="max-width:560px">
        <div id="add-msg"></div>
        <div class="form-group"><label>ID рецепта</label>
          <input id="f-id" placeholder="r009 (уникальный)"></div>
        <div class="form-group"><label>Название</label>
          <input id="f-name" placeholder="Борщ"></div>
        <div class="form-group"><label>Категория</label>
          <select id="f-cat">
            <option>Суп</option><option>Салат</option>
            <option>Десерт</option><option>Основное</option><option>Другое</option>
          </select></div>
        <div class="form-group"><label>Ингредиенты (через запятую)</label>
          <input id="f-ing" placeholder="морковь, лук, картофель"></div>
        <div class="form-group"><label>Шаги приготовления (каждый с новой строки)</label>
          <textarea id="f-steps" placeholder="Шаг 1&#10;Шаг 2"></textarea></div>
        <div class="form-group"><label>Время (мин)</label>
          <input id="f-time" type="number" min="1" value="30"></div>
        <div class="form-group"><label>Рейтинг (1-10)</label>
          <input id="f-rating" type="number" min="1" max="10" step="0.1" value="8.0"></div>
        <button class="btn btn-primary" onclick="addRecipe()">Добавить</button>
      </div>
    </div>

    <!-- SEARCH (изменённый блок) -->
    <div class="page" id="page-search">
      <div class="page-title">🔍 Поиск</div>
      <!-- Переключатель режима -->
      <div style="display:flex; gap:2rem; margin-bottom:1rem; align-items:center;">
        <label style="display:flex; align-items:center; gap:.3rem; cursor:pointer;">
          <input type="radio" name="search-mode" value="simple" checked onchange="switchSearchMode()"> Простой поиск
        </label>
        <label style="display:flex; align-items:center; gap:.3rem; cursor:pointer;">
          <input type="radio" name="search-mode" value="ingredients" onchange="switchSearchMode()"> По ингредиентам
        </label>
      </div>
      <!-- Простой поиск -->
      <div id="search-simple" style="display:flex; gap:.6rem; margin-bottom:1.2rem;">
        <input id="search-q" placeholder="Название, категория или ингредиент…"
               onkeydown="if(event.key==='Enter')doSimpleSearch()">
        <button class="btn btn-primary" onclick="doSimpleSearch()">Найти</button>
      </div>
      <!-- Поиск по ингредиентам -->
      <div id="search-ingredients" style="display:none;">
        <div class="search-bar" style="margin-bottom:.6rem;">
          <input id="search-ing-list" placeholder="Введите ингредиенты через запятую, например: курица, сливки"
                 onkeydown="if(event.key==='Enter')doIngredientSearch('AND')">
        </div>
        <div style="display:flex; gap:.6rem; margin-bottom:1.2rem;">
          <button class="btn btn-primary" onclick="doIngredientSearch('AND')">Найти (все ингредиенты)</button>
          <button class="btn btn-secondary" onclick="doIngredientSearch('OR')">Найти (любой ингредиент)</button>
        </div>
      </div>
      <div class="grid" id="search-results"></div>
    </div>

    <!-- REPORT -->
    <div class="page" id="page-report">
      <div class="page-title">📊 Отчёт (QuickSort)</div>
      <div class="card">
        <div class="row mb">
          <select id="rep-sort">
            <option value="rating">По рейтингу</option>
            <option value="name">По названию</option>
            <option value="time">По времени</option>
          </select>
          <select id="rep-rev">
            <option value="true">По убыванию</option>
            <option value="false">По возрастанию</option>
          </select>
          <button class="btn btn-primary" onclick="loadReport()">Сформировать</button>
        </div>
        <table>
          <thead><tr>
            <th>#</th><th>ID</th><th>Название</th><th>Категория</th>
            <th>Рейтинг</th><th>Время</th><th></th>
          </tr></thead>
          <tbody id="report-body"></tbody>
        </table>
      </div>
    </div>

    <!-- RELATIONS -->
    <div class="page" id="page-relations">
      <div class="page-title">🔗 Граф связей</div>
      <div class="card" style="max-width:500px;margin-bottom:1rem">
        <div id="rel-msg"></div>
        <div class="form-group"><label>ID рецепта 1</label>
          <input id="rel-id1" placeholder="r001"></div>
        <div class="form-group"><label>ID рецепта 2</label>
          <input id="rel-id2" placeholder="r003"></div>
        <div class="form-group"><label>Вес связи (сила похожести)</label>
          <input id="rel-w" type="number" step="0.1" value="1.0"></div>
        <div class="row">
          <button class="btn btn-primary" onclick="addRel()">Добавить связь</button>
          <button class="btn btn-danger" onclick="removeRel()">Удалить связь</button>
        </div>
      </div>
      <div class="card">
        <div style="font-weight:700;margin-bottom:.8rem">Все рёбра графа</div>
        <table>
          <thead><tr><th>Рецепт 1</th><th>Рецепт 2</th><th>Вес</th></tr></thead>
          <tbody id="edges-body"></tbody>
        </table>
      </div>
    </div>

    <!-- RECOMMEND -->
    <div class="page" id="page-recommend">
      <div class="page-title">💡 Рекомендации (BFS)</div>
      <div class="card" style="max-width:440px;margin-bottom:1rem">
        <div class="form-group"><label>ID рецепта</label>
          <input id="rec-id" placeholder="r001"></div>
        <div class="form-group"><label>Глубина BFS</label>
          <input id="rec-depth" type="number" min="1" max="5" value="2"></div>
        <button class="btn btn-primary" onclick="getRecommend()">Найти похожие</button>
      </div>
      <div class="grid" id="rec-results"></div>
    </div>

    <!-- DIJKSTRA -->
    <div class="page" id="page-dijkstra">
      <div class="page-title">📡 Алгоритм Дейкстры</div>
      <div class="card" style="max-width:440px;margin-bottom:1rem">
        <div class="form-group"><label>Стартовый рецепт</label>
          <input id="dijk-id" placeholder="r001"></div>
        <button class="btn btn-primary" onclick="getDijkstra()">Вычислить</button>
      </div>
      <div class="card">
        <div style="font-weight:700;margin-bottom:.8rem">Расстояния от стартового рецепта</div>
        <table>
          <thead><tr><th>Рецепт</th><th>Название</th><th>Расстояние (вес)</th></tr></thead>
          <tbody id="dijk-body"></tbody>
        </table>
      </div>
    </div>

    <!-- BENCH -->
    <div class="page" id="page-bench">
      <div class="page-title">⚡ Нагрузочное тестирование</div>
      <div class="card" style="max-width:440px;margin-bottom:1rem">
        <div class="form-group"><label>Количество записей N</label>
          <select id="bench-n">
            <option value="100">100</option>
            <option value="1000" selected>1 000</option>
            <option value="5000">5 000</option>
            <option value="10000">10 000</option>
          </select></div>
        <button class="btn btn-primary" onclick="runBench()" id="bench-btn">Запустить тест</button>
      </div>
      <div id="bench-results"></div>
    </div>

  </main>
</div>

<!-- Modal -->
<div class="modal-bg" id="modal-bg" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="flex-between mb"><h3 id="modal-title"></h3>
      <button class="btn btn-secondary btn-sm" onclick="closeModal()">✕</button></div>
    <div id="modal-body"></div>
  </div>
</div>

<script>
// ─── Navigation ───────────────────────────
document.querySelectorAll('.nav-item').forEach(el=>{
  el.addEventListener('click',()=>{
    document.querySelectorAll('.nav-item').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('page-'+el.dataset.page).classList.add('active');
    onPageOpen(el.dataset.page);
  });
});
function onPageOpen(p){
  if(p==='home')loadHome();
  if(p==='recipes')loadRecipes();
  if(p==='report')loadReport();
  if(p==='relations')loadEdges();
  if(p==='search')switchSearchMode(); // инициализация режима поиска
}

// ─── API helpers ──────────────────────────
async function api(path,opts={}){
  const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});
  return r.json();
}

// ─── Stars helper ─────────────────────────
function stars(r){
  const full=Math.round(r/2);
  return '★'.repeat(full)+'☆'.repeat(5-full);
}

// ─── Recipe card ──────────────────────────
function recipeCard(r,onclick=''){
  return `<div class="recipe-card" onclick="${onclick||`showRecipe('${r.id}')`}">
    <div class="rc-name">${r.name}</div>
    <div class="rc-cat">${r.category}</div>
    <div class="stars">${stars(r.rating)} ${r.rating}</div>
    <div class="rc-meta" style="display:flex;gap:.5rem;margin-top:.6rem;flex-wrap:wrap">
      <span class="badge badge-orange">⏱ ${r.time_min} мин</span>
      <span class="badge badge-purple">#${r.id}</span>
    </div>
  </div>`;
}

// ─── HOME ─────────────────────────────────
async function loadHome(){
  const d=await api('/api/stats');
  document.getElementById('home-stats').innerHTML=`
    <div class="stat-card"><div class="val">${d.total}</div><div class="lbl">Рецептов</div></div>
    <div class="stat-card"><div class="val">${d.categories}</div><div class="lbl">Категорий</div></div>
    <div class="stat-card"><div class="val">${d.avg_rating}</div><div class="lbl">Средний рейтинг</div></div>
    <div class="stat-card"><div class="val">${d.edges}</div><div class="lbl">Связей в графе</div></div>
  `;
  const top=await api('/api/recipes?sort=rating&reverse=true&limit=4');
  document.getElementById('home-top').innerHTML=`<div class="grid">${top.map(r=>recipeCard(r)).join('')}</div>`;
}

// ─── ALL RECIPES ──────────────────────────
async function loadRecipes(){
  const sort=document.getElementById('sort-by').value;
  const rev=sort==='rating'?'true':'false';
  const data=await api(`/api/recipes?sort=${sort}&reverse=${rev}`);
  document.getElementById('recipes-grid').innerHTML=data.map(r=>recipeCard(r)).join('');
}
document.getElementById('sort-by').onchange=loadRecipes;

// ─── ADD ──────────────────────────────────
async function addRecipe(){
  const body={
    id:document.getElementById('f-id').value.trim(),
    name:document.getElementById('f-name').value.trim(),
    category:document.getElementById('f-cat').value,
    ingredients:document.getElementById('f-ing').value.split(',').map(s=>s.trim()).filter(Boolean),
    steps:document.getElementById('f-steps').value.split('\n').map(s=>s.trim()).filter(Boolean),
    time_min:parseInt(document.getElementById('f-time').value),
    rating:parseFloat(document.getElementById('f-rating').value),
  };
  const res=await api('/api/recipes',{method:'POST',body:JSON.stringify(body)});
  const el=document.getElementById('add-msg');
  if(res.ok){
    el.innerHTML=`<div class="alert alert-ok">✅ Рецепт "${res.recipe.name}" добавлен!</div>`;
    ['f-id','f-name','f-ing','f-steps'].forEach(id=>document.getElementById(id).value='');
  } else {
    el.innerHTML=`<div class="alert alert-err">❌ ${res.error}</div>`;
  }
}

// ─── SEARCH (новые функции) ───────────────
function switchSearchMode() {
  const mode = document.querySelector('input[name="search-mode"]:checked').value;
  document.getElementById('search-simple').style.display = mode === 'simple' ? 'flex' : 'none';
  document.getElementById('search-ingredients').style.display = mode === 'ingredients' ? 'block' : 'none';
  document.getElementById('search-results').innerHTML = '';
}

async function doSimpleSearch(){
  const q = document.getElementById('search-q').value.trim();
  if (!q) return;
  const data = await api(`/api/search?q=${encodeURIComponent(q)}`);
  document.getElementById('search-results').innerHTML =
    data.length ? data.map(r => recipeCard(r)).join('')
    : '<div class="text-muted">Ничего не найдено.</div>';
}

async function doIngredientSearch(mode){
  const list = document.getElementById('search-ing-list').value.trim();
  if (!list) return;
  const data = await api(`/api/search/ingredients?ingredients=${encodeURIComponent(list)}&mode=${mode}`);
  document.getElementById('search-results').innerHTML =
    data.length ? data.map(r => recipeCard(r)).join('')
    : '<div class="text-muted">Ничего не найдено.</div>';
}

// ─── REPORT ───────────────────────────────
async function loadReport(){
  const sort=document.getElementById('rep-sort').value;
  const rev=document.getElementById('rep-rev').value;
  const data=await api(`/api/report?sort=${sort}&reverse=${rev}`);
  document.getElementById('report-body').innerHTML=data.map((r,i)=>`<tr>
    <td>${i+1}</td><td><span class="badge badge-purple">${r.id}</span></td>
    <td><a href="#" onclick="showRecipe('${r.id}');return false">${r.name}</a></td>
    <td>${r.category}</td>
    <td><span class="stars">${stars(r.rating)}</span> ${r.rating}</td>
    <td>${r.time_min} мин</td>
    <td><button class="btn btn-danger btn-sm" onclick="deleteRecipe('${r.id}')">🗑</button></td>
  </tr>`).join('');
}

// ─── RELATIONS ────────────────────────────
async function loadEdges(){
  const data=await api('/api/edges');
  document.getElementById('edges-body').innerHTML=data.map(e=>`<tr>
    <td>${e[0]}</td><td>${e[1]}</td><td>${e[2]}</td>
  </tr>`).join('');
}
async function addRel(){
  const body={id1:document.getElementById('rel-id1').value.trim(),
    id2:document.getElementById('rel-id2').value.trim(),
    weight:parseFloat(document.getElementById('rel-w').value)};
  const res=await api('/api/relations',{method:'POST',body:JSON.stringify(body)});
  document.getElementById('rel-msg').innerHTML=res.ok
    ?`<div class="alert alert-ok">✅ Связь добавлена</div>`
    :`<div class="alert alert-err">❌ ${res.error}</div>`;
  loadEdges();
}
async function removeRel(){
  const body={id1:document.getElementById('rel-id1').value.trim(),
    id2:document.getElementById('rel-id2').value.trim()};
  const res=await api('/api/relations',{method:'DELETE',body:JSON.stringify(body)});
  document.getElementById('rel-msg').innerHTML=res.ok
    ?`<div class="alert alert-ok">✅ Связь удалена</div>`
    :`<div class="alert alert-err">❌ ${res.error}</div>`;
  loadEdges();
}

// ─── RECOMMEND ────────────────────────────
async function getRecommend(){
  const rid=document.getElementById('rec-id').value.trim();
  const depth=document.getElementById('rec-depth').value;
  const data=await api(`/api/recommend/${rid}?depth=${depth}`);
  if(data.error){
    document.getElementById('rec-results').innerHTML=`<div class="alert alert-err">❌ ${data.error}</div>`;
    return;
  }
  document.getElementById('rec-results').innerHTML=
    data.length?data.map(r=>recipeCard(r)).join('')
    :'<div class="text-muted">Похожих рецептов не найдено (нет связей на данной глубине).</div>';
}

// ─── DIJKSTRA ─────────────────────────────
async function getDijkstra(){
  const rid=document.getElementById('dijk-id').value.trim();
  const data=await api(`/api/dijkstra/${rid}`);
  if(data.error){
    document.getElementById('dijk-body').innerHTML=`<tr><td colspan="3" style="color:var(--red)">${data.error}</td></tr>`;
    return;
  }
  document.getElementById('dijk-body').innerHTML=data.map(d=>`<tr>
    <td><span class="badge badge-purple">${d.id}</span></td>
    <td>${d.name}</td>
    <td><span class="badge badge-green">${d.distance}</span></td>
  </tr>`).join('');
}

// ─── BENCH ────────────────────────────────
async function runBench(){
  const n=document.getElementById('bench-n').value;
  const btn=document.getElementById('bench-btn');
  btn.textContent='⏳ Идёт тестирование…';btn.disabled=true;
  const data=await api(`/api/benchmark?n=${n}`);
  btn.textContent='Запустить тест';btn.disabled=false;
  document.getElementById('bench-results').innerHTML=`
  <div class="card">
    <table class="bench-table">
      <thead><tr><th>Операция</th><th>Среднее время (мкс)</th><th>Сложность</th></tr></thead>
      <tbody>
        <tr><td>Вставка (HashTable + 2×AVL + Graph)</td>
            <td><span class="badge badge-green">${data.insert_us}</span></td><td>O(log n)</td></tr>
        <tr><td>Поиск по ID (HashTable)</td>
            <td><span class="badge badge-green">${data.search_us}</span></td><td>O(1) avg</td></tr>
        <tr><td>Обход AVL-дерева (get sorted)</td>
            <td><span class="badge badge-orange">${data.sort_tree_us}</span></td><td>O(n)</td></tr>
        <tr><td>QuickSort (generate_report)</td>
            <td><span class="badge badge-orange">${data.quicksort_us}</span></td><td>O(n log n)</td></tr>
        <tr><td>BFS (глубина 2)</td>
            <td><span class="badge badge-green">${data.bfs_us}</span></td><td>O(V+E)</td></tr>
      </tbody>
    </table>
    <div class="text-muted mt">N = ${data.n} записей. Время вставки и поиска — среднее на одну операцию.</div>
  </div>`;
}

// ─── MODAL: Show recipe ───────────────────
async function showRecipe(id){
  const r=await api(`/api/recipes/${id}`);
  document.getElementById('modal-title').textContent=r.name;
  document.getElementById('modal-body').innerHTML=`
    <div class="row mb">
      <span class="badge badge-orange">${r.category}</span>
      <span class="badge badge-purple">#${r.id}</span>
      <span class="stars">${stars(r.rating)} ${r.rating}</span>
      <span class="badge badge-green">⏱ ${r.time_min} мин</span>
    </div>
    <div style="font-weight:600;margin-bottom:.4rem">Ингредиенты:</div>
    <div class="mb">${r.ingredients.map(i=>`<span class="chip">${i}</span>`).join('')}</div>
    <div style="font-weight:600;margin-bottom:.4rem">Приготовление:</div>
    <ol style="padding-left:1.2rem;line-height:1.8">
      ${r.steps.map(s=>`<li>${s}</li>`).join('')}
    </ol>
    <div class="row mt">
      <button class="btn btn-danger btn-sm" onclick="deleteRecipe('${r.id}');closeModal()">🗑 Удалить</button>
    </div>
  `;
  document.getElementById('modal-bg').classList.add('open');
}
function closeModal(){document.getElementById('modal-bg').classList.remove('open')}

// ─── DELETE ───────────────────────────────
async function deleteRecipe(id){
  if(!confirm(`Удалить рецепт "${id}"?`))return;
  const res=await api(`/api/recipes/${id}`,{method:'DELETE'});
  if(res.ok){loadRecipes();loadHome();}
  else alert('Ошибка: '+res.error);
}

// Init
loadHome();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# 7. REST API ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


# --- Stats ---

@app.route("/api/stats")
def api_stats():
    recipes = [r for r in dm.hash_table.values() if isinstance(r, Recipe)]
    cats = len({r.category for r in recipes})
    avg = round(sum(r.rating for r in recipes) / len(recipes), 2) if recipes else 0
    edges = len(dm.graph.all_edges())
    return jsonify({"total": len(recipes), "categories": cats,
                    "avg_rating": avg, "edges": edges})


# --- Recipes list / create ---

@app.route("/api/recipes", methods=["GET"])
def api_recipes():
    sort = request.args.get("sort", "rating")
    reverse = request.args.get("reverse", "true").lower() != "false"
    limit = request.args.get("limit", None)
    if sort == "rating":
        recipes = dm.get_sorted_by_rating(reverse=reverse)
    elif sort == "name":
        recipes = dm.get_sorted_by_name()
        if reverse:
            recipes = list(reversed(recipes))
    else:
        recipes = dm.generate_report(sort_by=sort, reverse=reverse)
    if limit:
        recipes = recipes[:int(limit)]
    return jsonify([r.to_dict() for r in recipes if r])


@app.route("/api/recipes", methods=["POST"])
def api_add_recipe():
    data = request.json or {}
    try:
        r = Recipe(
            recipe_id=data["id"],
            name=data["name"],
            category=data.get("category", "Другое"),
            ingredients=data.get("ingredients", []),
            steps=data.get("steps", []),
            time_min=int(data.get("time_min", 30)),
            rating=float(data.get("rating", 5.0)),
        )
        dm.add_recipe(r)
        return jsonify({"ok": True, "recipe": r.to_dict()})
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# --- Single recipe ---

@app.route("/api/recipes/<recipe_id>", methods=["GET"])
def api_get_recipe(recipe_id):
    r = dm.get_recipe(recipe_id)
    if not r:
        return jsonify({"error": "Рецепт не найден"}), 404
    return jsonify(r.to_dict())


@app.route("/api/recipes/<recipe_id>", methods=["PUT"])
def api_update_recipe(recipe_id):
    data = request.json or {}
    try:
        r = dm.update_recipe(recipe_id, **data)
        return jsonify({"ok": True, "recipe": r.to_dict()})
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/recipes/<recipe_id>", methods=["DELETE"])
def api_delete_recipe(recipe_id):
    try:
        dm.delete_recipe(recipe_id)
        return jsonify({"ok": True})
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 404


# --- Search (простой) ---

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify([r.to_dict() for r in dm.search(q)])


# --- Search by ingredients (новый endpoint) ---

@app.route("/api/search/ingredients")
def api_search_ingredients():
    ingredients_str = request.args.get("ingredients", "")
    mode = request.args.get("mode", "and").lower()
    if ingredients_str.strip() == "":
        return jsonify([])
    ingredients_list = [i.strip().lower() for i in ingredients_str.split(",") if i.strip()]
    recipes = dm.search_by_ingredients(ingredients_list, mode)
    return jsonify([r.to_dict() for r in recipes])


# --- Report ---

@app.route("/api/report")
def api_report():
    sort = request.args.get("sort", "rating")
    reverse = request.args.get("reverse", "true").lower() != "false"
    return jsonify([r.to_dict() for r in dm.generate_report(sort_by=sort, reverse=reverse)])


# --- Relations / Graph ---

@app.route("/api/edges")
def api_edges():
    return jsonify(dm.graph.all_edges())


@app.route("/api/relations", methods=["POST"])
def api_add_relation():
    data = request.json or {}
    try:
        dm.add_relation(data["id1"], data["id2"], float(data.get("weight", 1.0)))
        return jsonify({"ok": True})
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/relations", methods=["DELETE"])
def api_remove_relation():
    data = request.json or {}
    dm.remove_relation(data.get("id1", ""), data.get("id2", ""))
    return jsonify({"ok": True})


# --- Recommend (BFS) ---

@app.route("/api/recommend/<recipe_id>")
def api_recommend(recipe_id):
    depth = int(request.args.get("depth", 2))
    try:
        results = dm.recommend(recipe_id, max_depth=depth)
        return jsonify([r.to_dict() for r in results if r])
    except KeyError as e:
        return jsonify({"error": str(e)}), 404


# --- Dijkstra ---

@app.route("/api/dijkstra/<recipe_id>")
def api_dijkstra(recipe_id):
    if not dm.graph.has_node(recipe_id):
        return jsonify({"error": f"Рецепт '{recipe_id}' не найден в графе"}), 404
    closest = dm.closest(recipe_id)
    return jsonify([{"id": r.id, "name": r.name, "distance": round(d, 2)}
                    for r, d in closest])


# --- Benchmark ---

@app.route("/api/benchmark")
def api_benchmark():
    n = int(request.args.get("n", 1000))
    n = min(n, 50000)
    result = dm.benchmark(n)
    return jsonify(result)


# ─────────────────────────────────────────────
# 8. ТОЧКА ВХОДА
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🍽  Система управления рецептами — Лабораторная работа №8")
    print("=" * 60)
    print("  Структуры данных: HashTable | AVLTree (×2) | Graph")
    print("  Алгоритмы: QuickSort | BFS | Dijkstra")
    print("  Запуск сервера: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)