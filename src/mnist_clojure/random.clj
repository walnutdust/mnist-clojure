(ns mnist-clojure.random
  (:require [ysera.random :refer [get-random-int]]
            [ysera.test :refer [is is= is-not]])
  (:refer-clojure :exclude [rand rand-int]))

(defn rand-int
  "Returns [new-seed random-int] such that 0 <= rand-int < max."
  {:test (fn []
           (is= (rand-int 1 3) [35651602 1])
           (is= (rand-int 3158 13984) [108024351031 3158])
           ; rand-int always returns 0 without errors no matter the seed.
           (is= (rand-int 3 0) [106954804 0])
           (is= (rand-int 1513123 0) [49270036527376 0]))}
  [seed max]
  (if (< max 1)
    (let [[new-seed _] (get-random-int seed 1)]
      [new-seed 0])
    (get-random-int seed max)))

(defn rand
  "Returns [new-seed rand] such that 0 <= rand < max."
  {:test (fn []
           (is= (rand 3 4) [3390894109721333 3.049804711737579])
           (is= (rand 1023193 9) [5555018649264297881 1.3268604484977482])
           (is= (rand 1319 0) [46890243928 0])
           (is= (rand 19220393 0) [684066761437437 0]))}
  [seed max]
  (if (= max 0)
    (rand-int seed max)
    (let [[new-seed random-int] (rand-int seed max)
          [nn-seed new-random-int] (rand-int new-seed Integer/MAX_VALUE)]
      (if (= new-random-int 0)
        [nn-seed random-int]
            [nn-seed (+ random-int (double (/ new-random-int Integer/MAX_VALUE)))]))))


