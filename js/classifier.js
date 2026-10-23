/**
 * classifier.js - TensorFlow.js + MobileNet 추론 엔진
 * MobileNet v2를 활용하여 특징을 추출하고 전이학습 모델로 실시간 판별합니다.
 */

class ImageClassifier {
  constructor() {
    this.mobilenetModel = null;
    this.customModel = null;
    this.featureDim = null;
    this.isReady = false;
    this.modelSource = 'none'; // 'server', 'uploaded', 'none'
  }

  /**
   * 1. MobileNet v2 베이스 모델 로드
   */
  async loadMobileNet(onProgress) {
    try {
      if (onProgress) onProgress('MobileNet 베이스 모델 다운로드 중...');
      // MobileNet v2, alpha 1.0 (출력 feature 차원: 1280)
      this.mobilenetModel = await mobilenet.load({
        version: 2,
        alpha: 1.0
      });

      // feature 차원 동적 감지 (224x224 빈 캔버스 활용)
      const dummyCanvas = document.createElement('canvas');
      dummyCanvas.width = 224;
      dummyCanvas.height = 224;
      const dummyFeature = this.mobilenetModel.infer(dummyCanvas, true);
      this.featureDim = dummyFeature.shape[1];
      dummyFeature.dispose();

      return true;
    } catch (err) {
      console.error('MobileNet 로딩 오류:', err);
      throw new Error('MobileNet 베이스 모델을 불러오지 못했습니다: ' + err.message);
    }
  }

  /**
   * 2. 서버 경로(./model/model.json)에서 학습된 모델 로드
   */
  async loadModelFromUrl(url = './model/model.json') {
    try {
      this.customModel = await tf.loadLayersModel(url);
      this.isReady = true;
      this.modelSource = 'server';
      return { success: true, source: 'server' };
    } catch (err) {
      console.warn('서버 경로의 모델 로드 실패:', err.message);
      this.isReady = false;
      return { success: false, error: err.message };
    }
  }

  /**
   * 3. 사용자가 직접 업로드한 model.json + weights.bin 로드 (선택형 편의기능)
   */
  async loadModelFromFiles(jsonFile, weightsFile) {
    try {
      this.customModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      this.isReady = true;
      this.modelSource = 'uploaded';
      return { success: true, source: 'uploaded' };
    } catch (err) {
      console.error('사용자 모델 파일 로드 오류:', err);
      throw new Error('모델 파일 형식이 올바르지 않습니다: ' + err.message);
    }
  }

  /**
   * 4. 이미지 판별 수행 (MobileNet 특징 추출 -> 커스텀 분류기 예측)
   * @param {HTMLImageElement|HTMLCanvasElement} imgElement 
   * @returns {Promise<{realProb: number, aiProb: number, verdict: string, confidence: number}>}
   */
  async predict(imgElement) {
    if (!this.mobilenetModel) {
      throw new Error('MobileNet 모델이 아직 준비되지 않았습니다.');
    }
    if (!this.customModel) {
      throw new Error('분류 모델이 로드되지 않았습니다.');
    }

    return tf.tidy(() => {
      // 1) MobileNet으로 이미지 특징 벡터 추출 (embedding: true)
      const features = this.mobilenetModel.infer(imgElement, true);

      // 2) 전이학습 커스텀 분류기로 예측
      const prediction = this.customModel.predict(features);
      const probabilities = prediction.dataSync(); // [realProb, aiProb]

      // 클래스 0: Real(실제 사진), 클래스 1: AI(생성 이미지)
      const realScore = probabilities[0];
      const aiScore = probabilities[1];

      // 백분율 정밀 변환
      let realProb = Math.round(realScore * 1000) / 10;
      let aiProb = Math.round(aiScore * 1000) / 10;

      // 합계 100% 보정
      const sum = realProb + aiProb;
      if (sum > 0) {
        realProb = Math.round((realProb / sum) * 1000) / 10;
        aiProb = Math.round((100 - realProb) * 10) / 10;
      }

      const verdict = realProb >= aiProb ? 'real' : 'ai';
      const confidence = Math.max(realProb, aiProb);

      return {
        realProb,
        aiProb,
        verdict,
        confidence
      };
    });
  }
}

// 전역 싱글톤 인스턴스
window.imageClassifier = new ImageClassifier();
