(ns mnist-clojure.matrix
  (:require [ysera.error :refer [error]]
            [ysera.test :refer [is is= is-not error?]]))

; Assumes Row Major column ordering, and that matrices are two dimensions

(defn dot
  "Multiplies two vectors. Assumes both vectors are represented by collections in the same dimension."
  {:test (fn []
           (is= (dot '(1 2 3) '(1 2 3)) 14)
           ; Throws an error if dimensions of input matrices are not the same.
           (error? (dot '(1 2 3) '(1 2))))}
  [v1 v2]
  (if (= (count v1) (count v2))
    (->> (map * v1 v2)
         (apply +))
    (error "Vectors passed to dot need to have the same length")))

(defn shape
  "Returns a row major matrix's dimensions in [rows columns], with [nil nil] if input is not a matrix."
  {:test (fn []
           ; Correctly computes the dimensions of a matrix
           (is= (shape '((1 2 3) (3 4 3))) [2 3])
           ; Returns [nil nil] for invalid matrices - not regular, not a 2d list, and simply an integer
           (is= (shape '((1) (2 3))) [nil nil])
           (is= (shape '(1 2 3)) [nil nil])
           (is= (shape 1) [nil nil]))}
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

(defn can-multiply?
  "Checks if two matrices can be multiplied by each other."
  {:test (fn []
           ; Invalid as the matrix dimensions are not in the form [m n] [n r]
           (is-not (can-multiply? `((1 2 3)) `((1 2 3))))
           (is (can-multiply? `((1 2) (3 4) (5 6)) `((1 2) (3 4)))))}
  [m1 m2]
  (let [[_ m1c] (shape m1)
        [m2r _] (shape m2)]
    (and (= m1c m2r)
         (not= m1c nil))))

(defn row-maj<->col-maj
  "Converts a matrix from row major to column major and vice-versa."
  {:test (fn []
           (is= (row-maj<->col-maj `((1 2 3) (1 2 3))) `((1 1) (2 2) (3 3)))
           (is= (row-maj<->col-maj `((1 1) (2 2) (3 3))) `((1 2 3) (1 2 3)))
           (is= (row-maj<->col-maj `((1) (2))) `((1 2)))
           (is= (row-maj<->col-maj `((1 2))) `((1) (2))))}
  [m]
  (if (some? (shape m))
    (apply map list m)
    (error "row-maj<->col-maj was called with invalid input.")))


(defn multiply
  "Multiplies matrices."
  ;TODO extend to n matrices
  ;TODO increase efficiency - current n^3 algorithm
  {:test (fn []
           (is= (multiply `((1 2 3) (4 5 6)) `((1 2) (3 4) (5 6))) `((22 28) (49 64))))}
  [m1 m2]
  (let [m2' (row-maj<->col-maj m2)]
    (map (fn [m1x] (map (fn [m2y] (dot m1x m2y)) m2')) m1)))
