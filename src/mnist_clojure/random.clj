(ns mnist-clojure.random
  (:require [ysera.random :refer [get-random-int]]
            [ysera.test :refer [is is= is-not]])
  (:refer-clojure :exclude [rand rand-int]))

(defn get-new-seed
  "Returns a new seed"
  {:test (fn []
           (is= (get-new-seed 0) 1)
           (is= (get-new-seed 1) 35651602))}
  [seed]
  (let [[new-seed _] (get-random-int seed 1)]
    new-seed))

(defn rand-int
  "Returns [new-seed random-int] such that 0 <= rand-int < max."
  {:test (fn []
           (is= (rand-int 1 1) [35651602 0])
           (is= (rand-int 1 0.1) [35651602 0])
           (is= (rand-int 1 3) [35651602 1])
           ; rand-int always returns 0 without errors no matter the seed.
           (is= (rand-int 3 0) [106954804 0])
           (is= (rand-int -3 0) [94371886 0])
           )}
  [seed max]
  (if (< max 1)
    [(get-new-seed seed) 0]
    (get-random-int seed max)))

(defn rand
  "Returns [new-seed rand] such that 0 <= rand < max."
  {:test (fn []
           (is= (rand 1 0) [35651602 0])
           (is= (rand 1 1) [35651602 4.656612875245797E-10])
           (is= (rand 1 0.5) [35651602 2.3283064376228985E-10])
           (is= (rand 1 0.1) [35651602 4.656612875245797E-11]))}
  [seed max]
  (if (<= max 0)
    [(get-new-seed seed) 0]
    (let [[seed n] (rand-int seed Integer/MAX_VALUE)
          weight (double (/ n Integer/MAX_VALUE))]
      [seed (* max weight)])))
