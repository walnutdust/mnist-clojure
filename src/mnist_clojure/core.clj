(ns mnist-clojure.core
  (:require [clojure.java.io :as io]
            [ysera.error :refer [error]]
            [ysera.test :refer [is is= is-not]]
            [mnist-clojure.matrix :refer [multiply]]
            [mnist-clojure.random :refer [rand-int rand]])
  (:refer-clojure :exclude [rand rand-int]))

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
    {:data       (->> data
                      (drop 16)
                      (map byte->int)
                      (map (partial * (/ 1 255))))
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
                            (map (partial partition width))
                            (map (fn [x] (map vec x))))))))

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

(defn rand-d-vec
  "Creates a vector of specified length with random decimal values between 0 to max"
  {:test (fn []
           (is= (rand-d-vec 4 1 0.4)
                [1961823115700386051
                 [0.01660157088963388 0.5044600700514671 0.8178222714074991 0.1430590959000676]]))}
  [length seed max]
  (reduce (fn [[seed rand-vals] _]
            (let [[next-seed rand-val] (rand seed max)]
              [next-seed (conj rand-vals rand-val)]))
          [seed []]
          (range length)))

(defn rand-vec-max-mag
  "Creates a vector of specified length with random decimal values between -max to max."
  {:test (fn []
           (is= (rand-vec-max-mag 4 2 0)
                [-9197343212719499864 [0 0 0 0]])
           (is= (rand-vec-max-mag 3 4 23)
                [-7068052242903947273 [-18.93359371783845 13.63314510538855 -6.004336956424794]]))}
  [length seed max]
  (let [[new-seed rand-vec] (rand-d-vec length seed (* 2 max))]
    [new-seed (mapv (partial + (- max)) rand-vec)]))

(defn initialize-layer
  "Initializes one layer of the neural network. Note the + 1 adjusts for the bias"
  ; y = Wx
  {:test (fn []
           (is= (initialize-layer 1 3 1 0) [-8728512804673154413 [[0 0] [0 0] [0 0]]])
           (is= (initialize-layer 3 2 24 1)
                [7786394753034826687
                 [[-0.6015623093589966 -0.48321265703216776 0.3397951933274954 0.6701701863995613]
                  [-0.8371181152002505 0.5806762038640101 0.5576858765248609 -0.32256568051994106]]]))}
  [input-count output-count seed max]
  (reduce (fn [[seed rand-vecs] _]
            (let [[new-seed rand-vec] (rand-vec-max-mag (+ input-count 1) seed max)]
              [new-seed (conj rand-vecs rand-vec)]))
          [seed []]
          (range output-count)))

(defn train-once
  "Processes one cycle of training the layer on the input and output"
  {:test (fn []
           (is= (train-once [[1] [0.2] [0.1]]
                            [1 0 0]
                            [[0.1 0.2 0.1 0.2]
                             [0.1 0.1 0.2 0.2]]
                            0.5)
                [[0.3400053299222091 0.4400053299222091 0.14800106598444182 0.22400053299222092]
                 [-0.1400053299222091 -0.1400053299222091 0.1519989340155582 0.1759994670077791]]))}
  [input desired-output layer learning-factor]
  (let [input (-> input
                  (flatten)
                  (conj 1))                                 ; conj 1 for the bias, note that it is added to the front
        intermediate (->> (multiply layer input)
                          (flatten)
                          (map (fn [x] (Math/exp x))))
        total (apply + intermediate)
        actual-output (map (fn [x] (/ x total)) intermediate) ; softmax
        updates (->> actual-output
                     (map - desired-output)
                     (map (partial * learning-factor))
                     (vec))]
    (vec (for [n (range (count updates))
               :let [updated-row (->> input
                                      (map (partial * (nth updates n)))
                                      (mapv + (nth layer n)))]]
           updated-row))))

(defn output-vector
  "Creates the output vector given the label."
  {:test (fn []
           (is= (output-vector 1) [0 1 0 0 0 0 0 0 0 0])
           (is= (output-vector 4) [0 0 0 0 1 0 0 0 0 0]))}
  [label]
  (-> (repeat 10 0)
      (vec)
      (assoc (+ 0 label) 1)))

(defn train
  [images-path labels-path training-factor epoches seed]
  (println "Training the model...")
  (let [{images     :data
         width      :width
         height     :height
         num-images :num-images} (mnist->images (get-mnist-image-data images-path))
        {labels     :data
         num-labels :num-labels} (get-mnist-label-data labels-path)]
    (if (= num-images num-labels)
      (loop [layer (initialize-layer (* width height) 10 0.1 seed)
             curr-epoch 0]
        (if (= curr-epoch epoches)
          layer
          (recur
            (loop [images images
                   labels labels
                   layer layer
                   n 1]
              (if (empty? images)
                layer
                (recur (rest images)
                       (rest labels)
                       (train-once (first images) (output-vector (first labels)) layer training-factor)
                       (do (when (= 0 (mod n 100)) (println n "/" num-images))
                           (inc n)))))
            (inc curr-epoch))))
      (error "The number of images and labels do not match"))))

(defn predict
  "Predicts the output based on the trained layer and the input."
  {:test (fn []
           (is= (predict [[1 9 1]
                          [2 8 2]
                          [3 2 3]]
                         [1 1])
                1))}
  [layer input]
  (->> (multiply layer (conj (flatten input) 1))
       (flatten)
       (map-indexed vector)
       (apply max-key second)
       (first)))

(defn train-and-test
  "Trains the model and tests it."
  [train-imgs train-labels test-imgs test-labels]
  (let [{test-images :data
         num-images  :num-images} (mnist->images (get-mnist-image-data test-imgs))
        {test-labels :data
         num-labels  :num-labels} (get-mnist-label-data test-labels)
        model (train train-imgs train-labels 0.2 10 1)]
    (if (= num-images num-labels)
      ((do (println "Testing the model..."))
       (loop [hits 0
              test-images test-images
              test-labels test-labels
              n 1]
         (if (empty? test-images)
           [hits num-images]
           (do (when (= 0 (mod n 100)) (println n "/" num-images))
               (if (= (predict model (first test-images)) (first test-labels))
                 (recur (inc hits) (rest test-images) (rest test-labels) (inc n))
                 (recur hits (rest test-images) (rest test-labels) (inc n)))))))
      (error "The number of test images and test labels are not equal"))))

(comment (train-and-test "resources/train-images.idx3-ubyte"
                         "resources/train-labels.idx1-ubyte"
                         "resources/test-images.idx3-ubyte"
                         "resources/test-labels.idx1-ubyte"))




