from rest_framework import serializers

class SampleSerializer(serializers.Serializer):
    features_at10 = serializers.DictField(child=serializers.FloatField(), required=False)
    features_at15 = serializers.DictField(child=serializers.FloatField(), required=False)
    def validate(self, data):
        if not data.get('features_at10') and not data.get('features_at15'):
            raise serializers.ValidationError("features_at10 또는 features_at15 중 하나는 필요합니다.")
        return data

class PredictRequestSerializer(serializers.Serializer):
    sample = SampleSerializer(required=False)
    samples = serializers.ListField(child=SampleSerializer(), required=False)
    return_reversal = serializers.BooleanField(required=False, default=True)
    def validate(self, data):
        if not data.get('sample') and not data.get('samples'):
            raise serializers.ValidationError("sample 또는 samples 중 하나를 제공하세요.")
        return data
