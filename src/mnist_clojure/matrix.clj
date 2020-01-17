(ns mnist-clojure.matrix
  (:require [ysera.error :refer [error]]
            [ysera.test :refer [is is= is-not error?]]))

; Assumes Row Major column ordering, and that matrices are two dimensions

(defn dot
  "Multiplies two vectors. Assumes both vectors are represented by collections in the same dimension."
  {:test (fn []
           (is= (dot '(1 2 3) '(1 2 3)) 14)
           (error? (dot '(1 2 3) '(1 2))))}
  [v1 v2]
  (if (= (count v1) (count v2))
    (->> (map * v1 v2)
         (apply +))
    (error "Vectors passed to dot need to have the same length")))

(defn matrix-dimensions
  "Returns a row major matrix's dimensions in [rows columns], with [nil nil] if input is not a matrix."
  {:test (fn []
           (is= (matrix-dimensions '((1 2 3) (3 4 3))) [2 3])
           (is= (matrix-dimensions '((1) (2 3))) [nil nil])
           (is= (matrix-dimensions '(1 2 3)) [nil nil])
           (is= (matrix-dimensions 1) [nil nil]))}
  [m]
  (if (and (coll? m) (coll? (first m)))
    (let [rows (count m)
          cols (count (first m))]
      (as-> (map count m) $
            (remove (partial = cols) $)
            (if (empty? $)
              [rows cols]
              [nil nil])))
    [nil nil]))

(defn- is-matrix?
  "Checks if a given input is a matrix by checking if it conforms to a regular structure in two dimensions."
  ; TODO validation that none of the inputs are further collections.
  {:test (fn []
           (is (is-matrix? `((1 2 3) (4 5 6))))
           (is-not (is-matrix? `((1 2) (4 5 6)))))}
  [m]
  (not= (matrix-dimensions m) [nil nil]))

(defn- can-multiply?
  "Checks if two matrices can be multiplied by each other."
  {:test (fn []
           (is-not (can-multiply? `((1 2 3)) `((1 2 3))))
           (is (can-multiply? `((1 2) (3 4) (5 6)) `((1 2) (3 4)))))}
  [m1 m2]
  (and (= (second (matrix-dimensions m1))
          (first (matrix-dimensions m2)))
       (not= (first (matrix-dimensions m1)) nil)))

(defn- row-maj<->col-maj
  "Converts a matrix from row major to column major and vice-versa."
  {:test (fn []
           (is= (row-maj<->col-maj `((1 2 3) (1 2 3))) `((1 1) (2 2) (3 3)))
           (error? (row-maj<->col-maj `(1 2))))}
  [m]
  (if (is-matrix? m)
    (let [n (count (first m))
          flat-m (flatten m)]
      (for [x (range n)
            :let [y (take-nth n (drop x flat-m))]]
        y))
    (error "row-maj<->col-maj was called with invalid input.")))

(defn multiply
  "Multiplies matrices."
  ;TODO extend to n matrices
  ;TODO increase efficiency - current n^3 algorithm
  {:test (fn []
           (error? (multiply '((1 2 3) (4 5 6)) `((1 2 3) (3 4 5))))
           (is= (multiply '((1 2 3)) '((1) (2) (3))) `((14)))
           (is= (multiply `((1 2 3) (4 5 6)) `((1 2) (3 4) (5 6))) `((22 28) (49 64))))}
  [m1 m2]
  (if (can-multiply? m1 m2)
    (let [m2' (row-maj<->col-maj m2)]
      (map (fn [m1x] (map (fn [m2y] (dot m1x m2y)) m2')) m1))
    (error "Multiply called with two matrices of invalid dimensions")))