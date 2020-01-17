(ns mnist-clojure.core
  (:require [clojure.java.io :as io]
            [ysera.error :refer [error]]
            [ysera.test :refer [is is= is-not]]))

(def ignore (constantly nil))                               ; Helper function to ignore the output

(defn bytes->int
  "Converts a byte array into an integer."
  {:test (fn []
           (is= (bytes->int '(0 0 8 3)) 2051)
           (is= (bytes->int '(0 0 8 1)) 2049))}
  [bytes]
  (->> bytes
       (map (partial format "%02x"))
       (apply (partial str "0x"))
       read-string))

(defn byte->int
  "Hacky method of converting a single byte to int. This works because clojure reads in bytes as signed integers."
  {:test (fn []
           (is= (byte->int 3) 3)
           (is= (byte->int 0) 0)
           (is= (byte->int -3) 253))}
  [byte]
  (if (neg-int? byte)
    (+ 256 byte)
    byte))

(defn read-mnist-file [file-name]
  "Reads the idx[x]-ubyte format and parses it into a byte array"
  (let [file (io/file file-name)
        b-array (byte-array (.length file))]
    (with-open [stream (io/input-stream file)]
      (.read stream b-array))
    b-array))

(defn sublist-bytes->int
  "Converts the bytes between the start index of the coll to start+4 to an unsigned integer"
  {:test (fn []
           (is= (sublist-bytes->int '(1 2 3 0 0 8 3 5) 3 4) 2051))}
  [coll start length]
  (->> coll
       (drop start)
       (take length)
       (bytes->int)))

(defn get-mnist-data [file-name magic-number]
  "Reads a mnist data file and validates the magic number"
  (let [data (read-mnist-file file-name)]
    (if (= (sublist-bytes->int data 0 4) magic-number)
      data
      (error "Data failed validation"))))

(defn get-mnist-image-data [file-name]
  "Get mnist images data."
  (let [data (get-mnist-data file-name 2051)]
    {:data       (map byte->int (drop 16 data))
     :num-images (sublist-bytes->int data 4 4)
     :height     (sublist-bytes->int data 8 4)
     :width      (sublist-bytes->int data 12 4)}))

(defn get-mnist-label-data [file-name]
  "Get mnist label data."
  (let [data (get-mnist-data file-name 2049)]
    {:data       (map byte->int (drop 8 data))
     :num-labels (sublist-bytes->int data 4 4)}))

(defn mnist->images
  "Converts mnist data to images"
  {:test (fn []
           (is= (mnist->images {:data   '(2 5 4 6 9 4 2 3 5 0 2 7)
                                :width  2
                                :height 2})
                '(((2 5) (4 6))
                  ((9 4) (2 3))
                  ((5 0) (2 7)))))}
  [image-data]
  (let [{data   :data
         width  :width
         height :height} image-data]
    (->> data
         (partition (* width height))
         (map (partial partition width)))))

(defn sigmoid
  "Performs the sigmoid function on the input."
  {:test (fn []
           (is= (sigmoid 0) 0.5)
           (is= (sigmoid -1) 0.2689414213699951)
           (is= (sigmoid 1) 0.7310585786300049))}
  [input]
  (->> (- input)
       (Math/exp)
       (+ 1)
       (/ 1)))

