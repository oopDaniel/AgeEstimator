<template>
  <div class="photo-uploader">
    <h1>{{ msg }}</h1>
    <section>
      <div>Upload your image below</div>
      <div class="desc">(square image size preferred)</div>
      <section class="card-section center">
        <el-card class="card-box">
          <el-upload
            class="avatar-uploader"
            action=""
            :show-file-list="false"
            :on-preview="handleAvatarPreview"
            :on-success="handleAvatarSuccess"
            :before-upload="beforeAvatarUpload"
            :multiple="false"
          >
            <i class="el-icon-plus avatar-uploader-icon"></i>
          </el-upload>
        </el-card>
        <el-dialog :visible.sync="dialogVisible">
          <img width="100%" :src="imageUrl" alt="" />
        </el-dialog>
      </section>
    </section>
  </div>
</template>

<script>
const VALID_FILE_TYPES = ['image/png', 'image/jpeg']
const UNSUPPORTED_MSG = 'Unsupported picture format. Please upload a JPG or PNG'
const TOO_LARGE_MSG = 'Picture size can not exceed 2MB'
const MAX_SIZE = 1024 ** 2

export default {
  name: 'PhotoUploader',
  props: {
    msg: String
  },
  data: () => ({
    imageUrl: '',
    dialogVisible: false
  }),
  methods: {
    handleAvatarPreview(file) {
      this.imageUrl = file.url
      this.dialogVisible = true
    },
    handleAvatarSuccess() {
      console.log('TODO')
    },
    beforeAvatarUpload({ type, size }) {
      const isValidType = VALID_FILE_TYPES.includes(type)
      const isLt2M = size / MAX_SIZE < 2

      if (!isValidType) {
        this.$message.error(UNSUPPORTED_MSG)
      } else if (!isLt2M) {
        this.$message.error(TOO_LARGE_MSG)
      }

      return isValidType && isLt2M
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.photo-uploader {
  color: var(--title);
}

.desc {
  margin-top: 0.2rem;
  color: var(--desc);
  font-style: italic;
}

.card-section {
  margin-top: 2rem;
  flex-direction: column;
}
.card-box {
  max-width: 250px;
  box-shadow: 0 2px 6px 0 rgba(0, 0, 0, 0.1);
  &:hover {
    box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.1);
  }
}

.avatar-uploader .el-upload {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}
.avatar-uploader .el-upload:hover {
  border-color: #409eff;
}
.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  line-height: 178px;
  text-align: center;
}
.avatar {
  width: 178px;
  height: 178px;
  display: block;
}
</style>
