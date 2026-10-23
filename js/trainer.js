/**
 * trainer.js - TensorFlow.js + MobileNet 전이학습(Transfer Learning) 엔진
 * 실제 사진과 AI 생성 이미지를 MobileNet 특징 추출기를 거쳐 커스텀 헤드 모델로 학습시킵니다.
 */

class DatasetManager {
  constructor() {
    this.realImages = []; // { id, file, dataUrl, imgElement }
    this.aiImages = [];
    this.nextId = 1;
  }

  async addFiles(files, label) {
    const list = label === 'real' ? this.realImages : this.aiImages;
    const addedItems = [];

    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;

      const dataUrl = await this.readFileAsDataURL(file);
      const imgElement = await this.loadImageElement(dataUrl);

      const item = {
        id: this.nextId++,
        name: file.name,
        size: file.size,
        file,
        dataUrl,
        imgElement,
        label
      };

      list.push(item);
      addedItems.push(item);
    }

    return addedItems;
  }

  removeImage(id, label) {
    if (label === 'real') {
      this.realImages = this.realImages.filter(img => img.id !== id);
    } else {
      this.aiImages = this.aiImages.filter(img => img.id !== id);
    }
  }

  clear(label) {
    if (label === 'real') {
      this.realImages = [];
    } else if (label === 'ai') {
      this.aiImages = [];
    } else {
      this.realImages = [];
      this.aiImages = [];
    }
  }

  getCounts() {
    return {
      real: this.realImages.length,
      ai: this.aiImages.length,
      total: this.realImages.length + this.aiImages.length
    };
  }

  readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  loadImageElement(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
}

class TransferTrainer {
  constructor() {
    this.mobilenetModel = null;
    this.transferModel = null;
    this.featureDim = 1280;
    this.isTraining = false;
    this.stopRequested = false;
  }

  /**
   * MobileNet 베이스 모델 준비
   */
  async initMobileNet(onLog) {
    if (this.mobilenetModel) return true;
    if (onLog) onLog('MobileNet v2 베이스 모델 로드 중...', 'info');

    this.mobilenetModel = await mobilenet.load({
      version: 2,
      alpha: 1.0
    });

    const dummyCanvas = document.createElement('canvas');
    dummyCanvas.width = 224;
    dummyCanvas.height = 224;
    const dummyFeature = this.mobilenetModel.infer(dummyCanvas, true);
    this.featureDim = dummyFeature.shape[1];
    dummyFeature.dispose();

    if (onLog) onLog(`MobileNet 로드 완료 (특징 벡터 차원: ${this.featureDim})`, 'success');
    return true;
  }

  /**
   * 커스텀 전이학습 모델(Classification Head) 빌드
   */
  buildTransferModel(learningRate = 0.0005) {
    const model = tf.sequential();

    // 1) 고차원 특징(1280차원)을 압축하는 은닉층
    model.add(tf.layers.dense({
      inputShape: [this.featureDim],
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));

    // 2) 과적합 방지 드롭아웃
    model.add(tf.layers.dropout({ rate: 0.25 }));

    // 3) 세부 분류 은닉층
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    // 4) 최종 2개 클래스 분류 (0: Real, 1: AI)
    model.add(tf.layers.dense({
      units: 2,
      activation: 'softmax'
    }));

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    this.transferModel = model;
    return model;
  }

  /**
   * 전이학습 실행
   */
  async train({
    dataset,
    epochs = 20,
    batchSize = 8,
    learningRate = 0.0005,
    onEpochEnd,
    onProgress,
    onLog
  }) {
    if (this.isTraining) throw new Error('이미 학습이 진행 중입니다.');
    this.isTraining = true;
    this.stopRequested = false;

    try {
      await this.initMobileNet(onLog);

      const totalImages = dataset.realImages.length + dataset.aiImages.length;
      if (dataset.realImages.length === 0 || dataset.aiImages.length === 0) {
        throw new Error('실제 사진과 AI 생성 이미지 모두 최소 1장 이상 필요합니다.');
      }

      if (onLog) onLog(`데이터셋 특징 벡터 추출 시작 (총 ${totalImages}장)...`, 'info');

      // 1. 모든 이미지에 대해 MobileNet 특징 추출
      const features = [];
      const labels = [];
      let processed = 0;

      // 실제 사진 (라벨 0 -> [1, 0])
      for (const item of dataset.realImages) {
        if (this.stopRequested) break;
        const feat = tf.tidy(() => this.mobilenetModel.infer(item.imgElement, true));
        features.push(feat);
        labels.push(0);
        processed++;
        if (onProgress) onProgress({ phase: 'extracting', current: processed, total: totalImages });
      }

      // AI 생성 이미지 (라벨 1 -> [0, 1])
      for (const item of dataset.aiImages) {
        if (this.stopRequested) break;
        const feat = tf.tidy(() => this.mobilenetModel.infer(item.imgElement, true));
        features.push(feat);
        labels.push(1);
        processed++;
        if (onProgress) onProgress({ phase: 'extracting', current: processed, total: totalImages });
      }

      if (this.stopRequested) {
        features.forEach(f => f.dispose());
        if (onLog) onLog('학습이 사용자에 의해 중단되었습니다.', 'warn');
        this.isTraining = false;
        return null;
      }

      if (onLog) onLog('특징 벡터 텐서 변환 및 셔플링 중...', 'info');

      // 텐서 병합
      const xs = tf.concat(features, 0);
      features.forEach(f => f.dispose()); // 개별 텐서 메모리 해제

      const ys = tf.tidy(() => {
        const labelsTensor = tf.tensor1d(labels, 'int32');
        return tf.oneHot(labelsTensor, 2);
      });

      // 2. 모델 구조 생성 및 컴파일
      this.buildTransferModel(learningRate);
      if (onLog) onLog(`모델 빌드 완료 (Epochs: ${epochs}, Batch Size: ${batchSize}, LR: ${learningRate})`, 'info');

      // 3. 모델 훈련
      const validationSplit = totalImages >= 6 ? 0.2 : 0.0;

      const history = await this.transferModel.fit(xs, ys, {
        epochs,
        batchSize,
        validationSplit,
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if (onEpochEnd) onEpochEnd(epoch + 1, epochs, logs);
            if (this.stopRequested) {
              this.transferModel.stopTraining = true;
              if (onLog) onLog(`Epoch ${epoch + 1}에서 학습이 중지되었습니다.`, 'warn');
            }
          }
        }
      });

      // 훈련용 텐서 메모리 해제
      xs.dispose();
      ys.dispose();

      if (onLog) onLog('전이 학습이 성공적으로 완료되었습니다! 모델을 저장할 수 있습니다.', 'success');
      this.isTraining = false;
      return history;

    } catch (err) {
      this.isTraining = false;
      if (onLog) onLog(`학습 실패: ${err.message}`, 'error');
      throw err;
    }
  }

  stop() {
    this.stopRequested = true;
  }

  /**
   * 브라우저 파일 다운로드로 model.json 및 weights.bin 저장
   */
  async saveModel(downloadName = 'model') {
    if (!this.transferModel) {
      throw new Error('저장할 학습된 모델이 없습니다.');
    }
    // TensorFlow.js 내장 브라우저 다운로드 스킴
    return await this.transferModel.save(`downloads://${downloadName}`);
  }
}

// 전역 인스턴스
window.datasetManager = new DatasetManager();
window.transferTrainer = new TransferTrainer();
