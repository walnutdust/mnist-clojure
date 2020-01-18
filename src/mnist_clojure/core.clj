(ns mnist-clojure.core
  (:require [clojure.java.io :as io]
            [ysera.error :refer [error]]
            [ysera.test :refer [is is= is-not]]
            [mnist-clojure.matrix :refer [multiply]]))

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
           (is= (mnist->images {:data       '(2 5 4 6 9 4 2 3 5 0 2 7)
                                :width      2
                                :height     2
                                :num-images 3})
                {:data '(((2 5) (4 6)) ((9 4) (2 3)) ((5 0) (2 7))), :width 2, :height 2, :num-images 3}))}
  [image-data]
  (let [{width  :width
         height :height} image-data]
    (update image-data
            :data
            (fn [data] (->> (partition (* width height) data)
                            (map (partial partition width)))))))

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

(defn rand-vec
  "Creates a vector of specified length with random decimal values between 0 to max"
  [length max]
  (->> (partial rand max)
       (repeatedly)
       (take length)
       (vec)))

(defn rand-vec-max-mag
  "Creates a vector of specified length with random decimal values between -max to max"
  [length max]
  (->> (* 2 max)
       (rand-vec length)
       (map (partial + (- max)))))

(defn initialize-layer
  "Initializes one layer of the neural network. Note the + 1 adjusts for the bias"
  ; y = Wx
  {:test (fn []
           (is= (initialize-layer 1 3 0) '((0.0 0.0) (0.0 0.0) (0.0 0.0))))}
  [input-count output-count max]
  (->> (partial rand-vec-max-mag (+ input-count 1) (* 2 max))
       (repeatedly)
       (take output-count)
       (vec)))                                              ; Make vec for random access

(defn train-once
  "Processes one cycle of training the layer on the input and output"
  [input output layer learning-factor]
  (let [updates (->> (multiply layer (conj (flatten input) 1)) ; conj 1 for the bias
                     (flatten)
                     (map sigmoid)                          ; TODO other functions could replace this too
                     (map - output)
                     (map (partial * learning-factor)))]
    (for [n (range (count updates))
          :let [updated-row (map (partial * (+ 1 (nth updates n))) (nth layer n))]]
      updated-row)))

(defn output-vector
  "Creates the output vector given the label."
  ;TODO generalize?
  {:test (fn []
           (is= (output-vector 1) `(0 1 0 0 0 0 0 0 0 0))
           (is= (output-vector 4) `(0 0 0 0 1 0 0 0 0 0)))}
  [label]
  (-> (repeat 10 0)
      (vec)
      (assoc (+ 0 label) 1)))

(defn train
  [images-path labels-path]
  (println "Training the model...")
  (let [{images     :data
         width      :width
         height     :height
         num-images :num-images} (mnist->images (get-mnist-image-data images-path))
        {labels     :data
         num-labels :num-labels} (get-mnist-label-data labels-path)]
    (if (= num-images num-labels)
      (loop [images images
             labels labels
             layer (initialize-layer (* width height) 10 0.1)
             n 1]
        (if (empty? images)
          layer
          (recur (rest images)
                 (rest labels)
                 (train-once (first images) (output-vector (first labels)) layer 0.2)
                 (do (when (= 0 (mod n 100)) (println n "/" num-images))
                     (inc n)))))
      (error "The number of images and labels do not match"))))

(defn predict
  "Predicts the output based on the trained layer and the input."
  {:test (fn []
           (is= (predict `((1 9 1)
                           (2 8 2)
                           (3 2 3))
                         `(1 1))
                1))}
  [layer input]                                             ; TODO generalize
  (->> (multiply layer (conj (flatten input) 1))
       (flatten)
       (map-indexed vector)
       (apply max-key second)
       (first)))

(comment
  (let [model (train "resources/train-images.idx3-ubyte" "resources/train-labels.idx1-ubyte")
        {test-images :data
         width       :width
         height      :height
         num-images  :num-images} (mnist->images (get-mnist-image-data "resources/test-images.idx3-ubyte"))
        {test-labels :data
         num-labels  :num-labels} (get-mnist-label-data "resources/test-labels.idx1-ubyte")]
    (do (println "Testing the model..."))
    (loop [hits 0
           test-images test-images
           test-labels test-labels
           n 1]
      (if (empty? test-images)
        [hits num-images]
        (do (when (= 0 (mod n 100)) (println n "/" num-images))
            (if (= (predict model (first test-images)) (first test-labels))
              (recur (inc hits) (rest test-images) (rest test-labels) (inc n))
              (recur hits (rest test-images) (rest test-labels) (inc n))))))))